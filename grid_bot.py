#!/usr/bin/env python3
"""
Diamond Grid Bot v4
- 5 coins: BTC, ETH, SOL, XRP, ADA
- Stake automatisch op basis van total_inleg in state.json
- Stop-loss 5% per positie
- Bestaande posities lopen door, nieuwe op juiste stake
- Reserve = 10% van total_inleg, minimaal €100
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from dotenv import load_dotenv

LOG = logging.getLogger("grid_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Configuratie ────────────────────────────────────────────────────────────
GRID_COINS   = ["BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR"]
GRID_LEVELS  = 8
RANGE_PCT    = 3.0       # ±3% range
STOP_LOSS    = 5.0       # 5% stop-loss per positie
MAX_POSITIONS = 4        # max open posities per coin
LOOP_SLEEP   = 60        # seconden tussen checks
TAKER_FEE    = 0.25      # 0.25% Bitvavo fee

STATE_FILE  = "/var/data/grid_state.json"
TRADES_FILE = "/var/data/grid_transactions.csv"

# Schaling op basis van total_inleg
# (min_inleg, stake_per_trade, max_positions)
SCHALING = [
    (3500, 140, 4),
    (2500, 110, 4),
    (1500, 90,  4),
    (800,  60,  3),
    (0,    45,  3),
]


def get_stake_and_max(total_inleg: float):
    for min_inleg, stake, max_pos in SCHALING:
        if total_inleg >= min_inleg:
            return stake, max_pos
    return 45, 3


def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0,
                "paused": False, "pause_reason": "", "total_inleg": 1795.0}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"grids": {}, "pnl": 0.0, "trades": 0, "wins": 0,
                "paused": False, "pause_reason": "", "total_inleg": 1795.0}


def save_state(state: dict):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def log_trade(side: str, symbol: str, amount: float, price: float,
              cost: float, pnl: float = 0.0):
    Path(TRADES_FILE).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(TRADES_FILE).exists()
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts", "market", "side", "amount", "price",
                        "quote_amount", "pnl"])
        w.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            symbol, side,
            f"{amount:.8f}", f"{price:.8f}",
            f"{cost:.4f}", f"{pnl:.4f}",
        ])


def setup_grid(symbol: str, price: float, state: dict):
    """Initialiseer een grid voor een coin."""
    step = price * (RANGE_PCT / 100) / (GRID_LEVELS // 2)
    low  = price * (1 - RANGE_PCT / 100)
    high = price * (1 + RANGE_PCT / 100)
    levels = [low + i * step for i in range(GRID_LEVELS + 1)]
    state["grids"][symbol] = {
        "low": low,
        "high": high,
        "step": step,
        "levels": levels,
        "positions": {},
        "last_reset": datetime.now(timezone.utc).isoformat(),
    }
    LOG.info("GRID SETUP %s | prijs=%.4f | laag=%.4f | hoog=%.4f | stap=%.4f",
             symbol, price, low, high, step)


def try_buy(exchange, symbol: str, level_idx: int, level_price: float,
            stake: float, op_id: str, state: dict) -> bool:
    """Koop op een grid level."""
    grid = state["grids"][symbol]

    # Check: level al open?
    key = str(level_idx)
    if key in grid["positions"]:
        return False

    # Check: max posities bereikt?
    total_inleg = float(state.get("total_inleg", 1795))
    _, max_pos = get_stake_and_max(total_inleg)
    if len(grid["positions"]) >= max_pos:
        return False

    # Check: genoeg saldo?
    try:
        bal = exchange.fetch_balance()
        free = float((bal.get("free") or {}).get("EUR", 0))
        reserve = max(100.0, total_inleg * 0.10)
        if free - stake < reserve:
            LOG.info("SKIP %s | te weinig saldo (free=%.2f reserve=%.2f)",
                     symbol, free, reserve)
            return False
    except Exception as e:
        LOG.warning("Saldo check mislukt: %s", e)
        return False

    # Kopen
    try:
        amount = stake / level_price
        amount = float(exchange.amount_to_precision(symbol, amount))
        params = {"operatorId": op_id} if op_id else {}
        order  = exchange.create_order(symbol, "market", "buy", amount, None, params)
        actual_price = float(order.get("price") or order.get("average") or level_price)
        actual_cost  = float(order.get("cost") or stake)
        actual_amt   = float(order.get("filled") or amount)

        sell_at  = actual_price + grid["step"]
        stop_at  = actual_price * (1 - STOP_LOSS / 100)

        grid["positions"][key] = {
            "amount":     actual_amt,
            "buy_price":  actual_price,
            "buy_cost":   actual_cost,
            "sell_at":    sell_at,
            "stop_at":    stop_at,
            "level":      level_idx,
            "ts":         datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        log_trade("BUY", symbol, actual_amt, actual_price, actual_cost)
        LOG.info("KOOP %s | level=%s | prijs=%.4f | stake=%.2f EUR | verkoop@%.4f | stop@%.4f",
                 symbol, key, actual_price, actual_cost, sell_at, stop_at)
        return True

    except Exception as e:
        LOG.error("KOOP MISLUKT %s level %s: %s", symbol, level_idx, e)
        return False


def try_sell(exchange, symbol: str, key: str, pos: dict,
             current_price: float, op_id: str, state: dict, reason: str) -> bool:
    """Verkoop een positie."""
    try:
        amount = float(exchange.amount_to_precision(symbol, pos["amount"]))
        if amount <= 0:
            del state["grids"][symbol]["positions"][key]
            save_state(state)
            return False

        params = {"operatorId": op_id} if op_id else {}
        order  = exchange.create_order(symbol, "market", "sell", amount, None, params)
        sell_price = float(order.get("price") or order.get("average") or current_price)
        sell_rev   = sell_price * amount

        fee_buy  = pos["buy_cost"] * (TAKER_FEE / 100)
        fee_sell = sell_rev * (TAKER_FEE / 100)
        pnl      = sell_rev - fee_sell - pos["buy_cost"] - fee_buy

        state["trades"] += 1
        if pnl > 0:
            state["wins"] += 1
        state["pnl"] = round(state.get("pnl", 0) + pnl, 4)
        del state["grids"][symbol]["positions"][key]
        save_state(state)

        log_trade("SELL", symbol, amount, sell_price, sell_rev, pnl)
        LOG.info("VERKOOP %s | reden=%s | prijs=%.4f | pnl=%+.4f EUR",
                 symbol, reason, sell_price, pnl)
        return True

    except Exception as e:
        LOG.error("VERKOOP MISLUKT %s key %s: %s", symbol, key, e)
        return False


def manage_coin(exchange, symbol: str, op_id: str, state: dict):
    """Beheer grid voor één coin."""
    if state.get("paused", False):
        return

    grid = state["grids"].get(symbol)

    # Huidige prijs ophalen
    try:
        ticker = exchange.fetch_ticker(symbol)
        price  = float(ticker.get("last") or ticker.get("close") or 0)
        if price <= 0:
            return
    except Exception as e:
        LOG.warning("Prijs ophalen mislukt %s: %s", symbol, e)
        return

    # Grid aanmaken of resetten als prijs buiten range is
    if not grid:
        setup_grid(symbol, price, state)
        grid = state["grids"][symbol]
    elif price < grid["low"] * 0.97 or price > grid["high"] * 1.03:
        LOG.info("GRID RESET %s | prijs=%.4f buiten range %.4f-%.4f",
                 symbol, price, grid["low"], grid["high"])
        # Behoud bestaande posities, reset alleen de range
        old_positions = grid.get("positions", {})
        setup_grid(symbol, price, state)
        state["grids"][symbol]["positions"] = old_positions
        grid = state["grids"][symbol]

    # Bestaande posities controleren (verkopen of stop-loss)
    for key in list(grid["positions"].keys()):
        pos = grid["positions"][key]
        if price >= pos["sell_at"]:
            try_sell(exchange, symbol, key, pos, price, op_id, state, "take_profit")
            time.sleep(0.5)
        elif price <= pos["stop_at"]:
            try_sell(exchange, symbol, key, pos, price, op_id, state, "stop_loss")
            time.sleep(0.5)

    # Geen nieuwe aankopen zolang er nog gesyncte posities open staan (alle coins)
    all_synced_done = not any(
        p.get("ts", "").startswith("2026-06-30T06:00")
        for g in state.get("grids", {}).values()
        for p in g.get("positions", {}).values()
    )
    if not all_synced_done:
        LOG.info("SKIP KOOP %s | gesyncte posities nog open bij andere coins", symbol)
        return

    # Nieuwe aankopen op grid levels
    total_inleg = float(state.get("total_inleg", 1795))
    stake, _ = get_stake_and_max(total_inleg)
    levels = grid.get("levels", [])

    for i, level in enumerate(levels[:-1]):  # niet het hoogste level kopen
        # Koop als prijs binnen 0.6% van dit level is en level onder huidige prijs
        if abs(price - level) / level < 0.006 and price > level:
            try_buy(exchange, symbol, i, level, stake, op_id, state)
            time.sleep(0.3)


def main():
    load_dotenv()
    exchange = ccxt.bitvavo({
        "apiKey":  os.getenv("BITVAVO_API_KEY", "").strip(),
        "secret":  os.getenv("BITVAVO_API_SECRET", "").strip(),
        "enableRateLimit": True,
    })
    exchange.load_markets()
    op_id = os.getenv("BITVAVO_OPERATOR_ID", "").strip()

    state = load_state()

    # Zorg dat total_inleg altijd aanwezig is
    if "total_inleg" not in state:
        state["total_inleg"] = 1795.0
        save_state(state)

    total_inleg = float(state.get("total_inleg", 1795))
    stake, max_pos = get_stake_and_max(total_inleg)
    LOG.info("Diamond Grid Bot v4 gestart | total_inleg=%.2f EUR | stake=%d EUR | max_pos=%d",
             total_inleg, stake, max_pos)

    while True:
        try:
            state = load_state()  # herlaad state elke loop (agent kan pauzeren)
            total_inleg = float(state.get("total_inleg", 1795))
            stake, _ = get_stake_and_max(total_inleg)

            if state.get("paused", False):
                LOG.info("Bot gepauzeerd: %s", state.get("pause_reason", ""))
                time.sleep(LOOP_SLEEP)
                continue

            for symbol in GRID_COINS:
                manage_coin(exchange, symbol, op_id, state)
                time.sleep(1)

            LOG.info("Loop klaar | total_inleg=%.2f | stake=%.0f | pnl=%+.2f | trades=%d",
                     total_inleg, stake,
                     state.get("pnl", 0), state.get("trades", 0))

        except Exception as e:
            LOG.error("Loop fout: %s", e)

        time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    main()
