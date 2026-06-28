#!/usr/bin/env python3
"""
Diamond Grid Bot v4
- 5 coins: BTC, ETH, SOL, XRP, ADA
- Automatische schaling op basis van saldo
- Marktfilter op BTC trend
- Stop-loss per positie
- Pauze systeem via state
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

GRID_COINS   = ["BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR"]
GRID_LEVELS  = 8
RANGE_PCT    = 3.0
LOOP_SLEEP   = 60
TAKER_FEE_PCT = 0.25
STOP_LOSS_PCT = 5.0

# Automatische schaling op basis van saldo
SCALING = [
    {"min": 0,     "max": 1000,  "stake": 30,  "max_pos": 2},
    {"min": 1000,  "max": 2000,  "stake": 60,  "max_pos": 3},
    {"min": 2000,  "max": 3000,  "stake": 90,  "max_pos": 4},
    {"min": 3000,  "max": 4000,  "stake": 110, "max_pos": 4},
    {"min": 4000,  "max": 99999, "stake": 140, "max_pos": 5},
]
RESERVE_PCT = 20  # altijd 20% van saldo als reserve


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_state() -> Dict[str, Any]:
    if not Path(STATE_FILE).exists():
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0, "paused": False, "pause_reason": ""}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0, "paused": False, "pause_reason": ""}


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
        self._balance_cache = {}
        self._balance_ts = 0.0

    def params(self):
        return {"operatorId": self.operator_id} if self.operator_id else {}

    def fetch_balance(self) -> Dict:
        # Cache saldo voor 5 minuten
        if time.time() - self._balance_ts < 300:
            return self._balance_cache
        try:
            bal = self.exchange.fetch_balance()
            self._balance_cache = bal
            self._balance_ts = time.time()
            return bal
        except Exception as e:
            LOG.warning("Saldo ophalen mislukt: %s", e)
            return self._balance_cache

    def free_eur(self) -> float:
        bal = self.fetch_balance()
        return float((bal.get("free") or {}).get("EUR", 0))

    def total_eur_value(self) -> float:
        """Schat totale portfolio waarde in EUR."""
        bal = self.fetch_balance()
        total = float((bal.get("free") or {}).get("EUR", 0))
        total += float((bal.get("used") or {}).get("EUR", 0))
        return total

    def get_scaling(self) -> Dict:
        """Bepaal stake en max posities op basis van saldo."""
        saldo = self.free_eur()
        for s in SCALING:
            if s["min"] <= saldo < s["max"]:
                return s
        return SCALING[-1]

    def reserve(self) -> float:
        return self.free_eur() * (RESERVE_PCT / 100)

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

    def btc_trend_ok(self) -> bool:
        """Check of BTC in uptrend zit — marktfilter."""
        try:
            candles = self.exchange.fetch_ohlcv("BTC/EUR", "1h", limit=24)
            if len(candles) < 24:
                return True
            prices = [c[4] for c in candles]  # close prijzen
            ma_short = sum(prices[-6:]) / 6    # 6-uur MA
            ma_long  = sum(prices[-24:]) / 24  # 24-uur MA
            trend_ok = ma_short >= ma_long * 0.98  # max 2% onder lange MA
            if not trend_ok:
                LOG.info("MARKTFILTER: BTC trend neerwaarts (ma_short=%.0f ma_long=%.0f)", ma_short, ma_long)
            return trend_ok
        except Exception as e:
            LOG.warning("BTC trend check mislukt: %s", e)
            return True  # bij twijfel doorgaan

    def setup_grid(self, symbol: str) -> Dict:
        p = self.price(symbol)
        low  = p * (1 - RANGE_PCT / 100)
        high = p * (1 + RANGE_PCT / 100)
        step = (high - low) / GRID_LEVELS
        levels = [round(low + i * step, 8) for i in range(GRID_LEVELS + 1)]
        scaling = self.get_scaling()

        LOG.info("GRID SETUP %s | prijs=%.4f | range=%.4f-%.4f | stap=%.4f | stake=%.0f EUR",
                 symbol, p, low, high, step, scaling["stake"])

        return {
            "symbol": symbol,
            "start_price": p,
            "low": low, "high": high, "step": step,
            "levels": levels,
            "positions": {},
            "created_at": now_iso(),
        }

    def do_buy(self, symbol: str, grid: Dict, level_idx: int) -> bool:
        level_key = str(level_idx)
        if level_key in grid["positions"]:
            return False

        scaling = self.get_scaling()
        max_pos = scaling["max_pos"]
        stake   = scaling["stake"]

        if len(grid["positions"]) >= max_pos:
            return False

        # Saldo check
        free = self.free_eur()
        reserve = free * (RESERVE_PCT / 100)
        if free - stake < reserve:
            LOG.debug("Onvoldoende saldo %s: %.2f EUR vrij", symbol, free)
            return False

        min_not = self.min_order(symbol)
        if stake < min_not:
            stake = min_not * 1.1

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
            sell_at    = buy_price + grid["step"]
            stop_at    = buy_price * (1 - STOP_LOSS_PCT / 100)

            grid["positions"][level_key] = {
                "amount":    buy_amount,
                "buy_price": buy_price,
                "buy_cost":  buy_cost,
                "sell_at":   sell_at,
                "stop_at":   stop_at,
            }
            self._balance_ts = 0  # reset cache
            save_state(self.state)

            append_csv({
                "ts": now_iso(), "market": symbol, "side": "BUY",
                "price": round(buy_price, 8), "amount": buy_amount,
                "quote": round(buy_cost, 4), "pnl": "",
            })
            LOG.info("KOOP %s | level=%s | prijs=%.6f | cost=%.2f EUR | verkoop=%.6f | stop=%.6f",
                     symbol, level_idx, buy_price, buy_cost, sell_at, stop_at)
            return True

        except Exception as e:
            LOG.warning("KOOP mislukt %s level %s: %s", symbol, level_idx, e)
            return False

    def do_sell(self, symbol: str, grid: Dict, level_key: str, reason: str = "take_profit") -> bool:
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
            self._balance_ts = 0
            save_state(self.state)

            append_csv({
                "ts": now_iso(), "market": symbol, "side": "SELL",
                "price": round(sell_price, 8), "amount": sell_amount,
                "quote": round(sell_rev, 4), "pnl": round(pnl, 4),
            })
            LOG.info("VERKOOP %s | level=%s | reden=%s | prijs=%.6f | pnl=%.4f EUR | totaal=%.2f EUR",
                     symbol, level_key, reason, sell_price, pnl, self.state["pnl"])
            return True

        except Exception as e:
            LOG.warning("VERKOOP mislukt %s level %s: %s", symbol, level_key, e)
            return False

    def manage(self, symbol: str):
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
            for lk in list(grid["positions"].keys()):
                self.do_sell(symbol, grid, lk, "reset")
            self.state["grids"][symbol] = self.setup_grid(symbol)
            save_state(self.state)
            return

        levels = grid["levels"]

        # Bepaal huidig level
        current_level = 0
        for i in range(len(levels) - 1):
            if levels[i] <= p < levels[i + 1]:
                current_level = i
                break

        # VERKOOP: take profit of stop-loss
        for lk, pos in list(grid["positions"].items()):
            if p >= pos["sell_at"]:
                self.do_sell(symbol, grid, lk, "take_profit")
            elif p <= pos["stop_at"]:
                LOG.warning("STOP-LOSS %s level %s | koop=%.4f stop=%.4f huidig=%.4f",
                            symbol, lk, pos["buy_price"], pos["stop_at"], p)
                self.do_sell(symbol, grid, lk, "stop_loss")

        # KOOP: alleen als marktfilter OK en prijs in onderste helft level
        if not self.state.get("paused", False):
            level_bottom = levels[current_level]
            level_top    = levels[current_level + 1] if current_level + 1 < len(levels) else p
            level_mid    = (level_bottom + level_top) / 2

            if p <= level_mid:
                self.do_buy(symbol, grid, current_level)

    def run(self):
        LOG.info("Grid Bot v4 gestart | coins=%s | %s levels | ±%.0f%% range",
                 GRID_COINS, GRID_LEVELS, RANGE_PCT)

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
                # Check pauze
                if self.state.get("paused"):
                    LOG.info("BOT GEPAUZEERD | reden=%s", self.state.get("pause_reason", "onbekend"))
                    time.sleep(LOOP_SLEEP)
                    continue

                # Check BTC trend elke 10 loops
                if loop % 10 == 0:
                    if not self.btc_trend_ok():
                        LOG.info("MARKTFILTER actief — geen nieuwe aankopen")

                for symbol in GRID_COINS:
                    self.manage(symbol)
                    time.sleep(1)

                loop += 1
                if loop % 10 == 0:
                    t = self.state.get("trades", 0)
                    w = self.state.get("wins", 0)
                    scaling = self.get_scaling()
                    LOG.info("STATUS | trades=%s | winrate=%.1f%% | pnl=%.2f EUR | stake=%.0f EUR | saldo=%.2f EUR",
                             t, (w / t * 100) if t else 0,
                             self.state.get("pnl", 0.0),
                             scaling["stake"], self.free_eur())

            except Exception as e:
                LOG.exception("Loop fout: %s", e)

            time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    bot = GridBot()
    bot.run()
