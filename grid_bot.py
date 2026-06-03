#!/usr/bin/env python3
"""
Diamond Grid Bot v3 - simpel en correct
Logica:
- Verdeel €75 over 10 levels per coin
- Koop 1 level per keer als prijs daalt
- Verkoop als prijs terugkomt boven aankoopprijs + 1 grid stap
- Nooit meer dan 1 open positie per level
- Wacht 60 seconden tussen elke check
"""
import json
import logging
import os
import time
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import ccxt
from dotenv import load_dotenv

LOG = logging.getLogger("grid_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

STATE_FILE  = "/opt/render/project/src/grid_state.json"
TRADES_FILE = "/opt/render/project/src/grid_transactions.csv"

GRID_COINS     = ["BTC/EUR", "ETH/EUR", "SOL/EUR"]
STAKE_PER_COIN = 45.0   # €45 per trade
GRID_LEVELS    = 8      # aantal levels
RANGE_PCT      = 3.0    # ±3% range
EUR_RESERVE    = 75.0   # altijd €75 vrij houden
LOOP_SLEEP     = 60     # seconden tussen checks
TAKER_FEE_PCT  = 0.25
MAX_POSITIONS  = 3      # max open posities per coin tegelijk


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_state() -> Dict[str, Any]:
    if not Path(STATE_FILE).exists():
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0}


def save_state(state):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def append_csv(row):
    Path(TRADES_FILE).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(TRADES_FILE).exists()
    cols = ["ts", "market", "side", "price", "amount", "quote", "pnl"]
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


class GridBot:
    def __init__(self):
        load_dotenv()
        self.exchange = ccxt.bitvavo({
            "apiKey":    os.getenv("BITVAVO_API_KEY", "").strip(),
            "secret":    os.getenv("BITVAVO_API_SECRET", "").strip(),
            "enableRateLimit": True,
        })
        self.operator_id = os.getenv("BITVAVO_OPERATOR_ID", "").strip()
        self.exchange.load_markets()
        self.state = load_state()

    def params(self):
        return {"operatorId": self.operator_id} if self.operator_id else {}

    def price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        return float(t.get("last") or 0)

    def min_order(self, symbol: str) -> float:
        m = self.exchange.market(symbol)
        c = (((m.get("limits") or {}).get("cost") or {}).get("min"))
        if c:
            return float(c)
        raw = (m.get("info") or {}).get("minOrderInQuoteAsset")
        return float(raw) if raw else 5.0

    def setup_grid(self, symbol: str) -> Dict:
        """Maak nieuwe grid aan op basis van huidige prijs."""
        p = self.price(symbol)
        low  = p * (1 - RANGE_PCT / 100)
        high = p * (1 + RANGE_PCT / 100)
        step = (high - low) / GRID_LEVELS
        levels = [round(low + i * step, 8) for i in range(GRID_LEVELS + 1)]
        stake_per_level = STAKE_PER_COIN  # €125 per trade, niet gedeeld door levels

        LOG.info("GRID SETUP %s | prijs=%.4f | laag=%.4f | hoog=%.4f | stap=%.4f | per_level=%.2f EUR",
                 symbol, p, low, high, step, stake_per_level)

        return {
            "symbol":           symbol,
            "start_price":      p,
            "low":              low,
            "high":             high,
            "step":             step,
            "levels":           levels,
            "stake_per_level":  stake_per_level,
            "positions":        {},   # "level_idx" -> {amount, buy_price, buy_cost, sell_at}
            "last_buy_level":   None, # voorkomt dubbele kopen op zelfde level
            "created_at":       now_iso(),
        }

    def do_buy(self, symbol: str, grid: Dict, level_idx: int) -> bool:
        """Koop op een level. Geeft True terug bij succes."""
        level_key = str(level_idx)

        # Nooit twee keer zelfde level kopen
        if level_key in grid["positions"]:
            return False

        # Max posities check
        if len(grid["positions"]) >= MAX_POSITIONS:
            return False

        stake = grid["stake_per_level"]
        min_not = self.min_order(symbol)
        if stake < min_not:
            stake = min_not * 1.1

        # Check of er genoeg saldo is inclusief reserve
        try:
            balance = self.exchange.fetch_balance()
            free_eur = float((balance.get("free") or {}).get("EUR", 0))
            if free_eur < stake + EUR_RESERVE:
                LOG.debug("Onvoldoende saldo voor %s: %.2f EUR beschikbaar", symbol, free_eur)
                return False
        except Exception:
            pass

        try:
            current_price = self.price(symbol)
            amount = stake / current_price
            amount_f = float(self.exchange.amount_to_precision(symbol, amount))
            if amount_f <= 0:
                return False

            order = self.exchange.create_order(
                symbol, "market", "buy", amount_f, None, self.params()
            )
            buy_price  = float(order.get("average") or order.get("price") or current_price)
            buy_amount = float(order.get("filled") or order.get("amount") or amount_f)
            buy_cost   = float(order.get("cost") or buy_amount * buy_price)
            sell_at    = buy_price + grid["step"]  # verkoop 1 stap hoger

            grid["positions"][level_key] = {
                "amount":    buy_amount,
                "buy_price": buy_price,
                "buy_cost":  buy_cost,
                "sell_at":   sell_at,
            }
            grid["last_buy_level"] = level_idx
            save_state(self.state)

            append_csv({
                "ts": now_iso(), "market": symbol, "side": "BUY",
                "price": round(buy_price, 8), "amount": buy_amount,
                "quote": round(buy_cost, 4), "pnl": "",
            })
            LOG.info("KOOP %s | level=%s | prijs=%.6f | amount=%.6f | cost=%.2f EUR | verkoop_bij=%.6f",
                     symbol, level_idx, buy_price, buy_amount, buy_cost, sell_at)
            return True

        except Exception as e:
            LOG.warning("KOOP mislukt %s level %s: %s", symbol, level_idx, e)
            return False

    def do_sell(self, symbol: str, grid: Dict, level_key: str) -> bool:
        """Verkoop een positie. Geeft True terug bij succes."""
        pos = grid["positions"].get(level_key)
        if not pos:
            return False

        try:
            amount_f = float(self.exchange.amount_to_precision(symbol, pos["amount"]))
            if amount_f <= 0:
                return False

            order = self.exchange.create_order(
                symbol, "market", "sell", amount_f, None, self.params()
            )
            sell_price  = float(order.get("average") or order.get("price") or pos["sell_at"])
            sell_amount = float(order.get("filled") or order.get("amount") or amount_f)
            sell_rev    = float(order.get("cost") or sell_amount * sell_price)
            fee_buy     = pos["buy_cost"] * (TAKER_FEE_PCT / 100)
            fee_sell    = sell_rev * (TAKER_FEE_PCT / 100)
            pnl         = sell_rev - fee_sell - pos["buy_cost"] - fee_buy

            self.state["pnl"]    = round(self.state.get("pnl", 0.0) + pnl, 4)
            self.state["trades"] = self.state.get("trades", 0) + 1
            if pnl > 0:
                self.state["wins"] = self.state.get("wins", 0) + 1

            del grid["positions"][level_key]
            save_state(self.state)

            append_csv({
                "ts": now_iso(), "market": symbol, "side": "SELL",
                "price": round(sell_price, 8), "amount": sell_amount,
                "quote": round(sell_rev, 4), "pnl": round(pnl, 4),
            })
            LOG.info("VERKOOP %s | level=%s | prijs=%.6f | pnl=%.4f EUR | totaal_pnl=%.2f EUR",
                     symbol, level_key, sell_price, pnl, self.state["pnl"])
            return True

        except Exception as e:
            LOG.warning("VERKOOP mislukt %s level %s: %s", symbol, level_key, e)
            return False

    def manage(self, symbol: str):
        """Beheer grid voor één coin."""
        grid = self.state["grids"].get(symbol)
        if not grid:
            return

        try:
            p = self.price(symbol)
        except Exception as e:
            LOG.warning("Prijs ophalen mislukt %s: %s", symbol, e)
            return

        low  = grid["low"]
        high = grid["high"]

        # Reset als prijs ver buiten range
        if p < low * 0.95 or p > high * 1.05:
            LOG.info("RESET %s | prijs=%.4f buiten range %.4f-%.4f", symbol, p, low, high)
            # Verkoop alle open posities eerst
            for lk in list(grid["positions"].keys()):
                self.do_sell(symbol, grid, lk)
            self.state["grids"][symbol] = self.setup_grid(symbol)
            save_state(self.state)
            return

        levels = grid["levels"]

        # Bepaal huidig level (tussen welke twee levels de prijs zit)
        current_level = 0
        for i in range(len(levels) - 1):
            if levels[i] <= p < levels[i + 1]:
                current_level = i
                break

        # VERKOOP: check alle open posities of sell_at bereikt is of stop-loss geraakt
        stop_loss_pct = 8.0
        for lk, pos in list(grid["positions"].items()):
            buy_price = pos["buy_price"]
            stop_price = buy_price * (1 - stop_loss_pct / 100)
            if p >= pos["sell_at"]:
                self.do_sell(symbol, grid, lk)
            elif p <= stop_price:
                LOG.warning("STOP-LOSS %s level %s | koop=%.4f stop=%.4f huidig=%.4f",
                            symbol, lk, buy_price, stop_price, p)
                self.do_sell(symbol, grid, lk)

        # KOOP: koop op het huidige level als er nog geen positie is
        # Alleen kopen als prijs dichter bij de onderkant van het level is
        level_bottom = levels[current_level]
        level_top    = levels[current_level + 1] if current_level + 1 < len(levels) else p
        level_mid    = (level_bottom + level_top) / 2

        # Koop als prijs in de onderste helft van het level zit
        if p <= level_mid:
            self.do_buy(symbol, grid, current_level)

    def run(self):
        LOG.info("Grid Bot v3 gestart | coins=%s | €%.0f/coin | %s levels",
                 GRID_COINS, STAKE_PER_COIN, GRID_LEVELS)

        # Setup grids
        for symbol in GRID_COINS:
            if symbol not in self.state.get("grids", {}):
                try:
                    self.state.setdefault("grids", {})[symbol] = self.setup_grid(symbol)
                    save_state(self.state)
                    time.sleep(2)
                except Exception as e:
                    LOG.error("Setup mislukt %s: %s", symbol, e)

        loop = 0
        while True:
            try:
                for symbol in GRID_COINS:
                    self.manage(symbol)
                    time.sleep(2)  # pauze tussen coins

                loop += 1
                if loop % 10 == 0:
                    t = self.state.get("trades", 0)
                    w = self.state.get("wins", 0)
                    LOG.info("STATUS | trades=%s | winrate=%.1f%% | pnl=%.2f EUR",
                             t, (w / t * 100) if t else 0, self.state.get("pnl", 0.0))

            except Exception as e:
                LOG.exception("Loop fout: %s", e)

            time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    bot = GridBot()
    bot.run()
