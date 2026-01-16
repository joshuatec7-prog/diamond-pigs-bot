#!/usr/bin/env python3
"""
Diamant-Pigs stijl Bitvavo bot (spot) met:
- Uptrend trend-follow (SMA + RSI)
- Downtrend mean-reversion (RSI oversold -> bounce), met lagere inzet
- Netto PnL (fees meegerekend)
- Extra logging + stats in staat.json
- Bestanden:
  - metFig.yaml
  - staat.json
  - transacties.csv

ENV:
- BITVAVO_API_KEY
- BITVAVO_API_SECRET
- BITVAVO_OPERATOR_ID  (vereist door ccxt Bitvavo createOrder)
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
import ccxt

# =========================================================
# Indicatoren
# =========================================================


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(length).mean()
    roll_down = down.rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))


# =========================================================
# Paden / helpers
# =========================================================

ROOT = Path(__file__).parent
CFG_FILE = ROOT / "metFig.yaml"
STATE_FILE = ROOT / "staat.json"
TRADES_FILE = ROOT / "transacties.csv"


def now_ts() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def load_cfg() -> dict:
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    else:
        state = {"positions": {}, "cooldown": {}, "pnl_eur": 0.0}

    state.setdefault("positions", {})
    state.setdefault("cooldown", {})
    state.setdefault("pnl_eur", 0.0)
    state.setdefault("stats", {})
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def append_trade(row: dict) -> None:
    hdr = not TRADES_FILE.exists()
    df = pd.DataFrame([row])
    df.to_csv(TRADES_FILE, index=False, mode="a", header=hdr)


def pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0 if b else 0.0


def update_stats_after_trade(state: dict, realized: float) -> None:
    stats = state.setdefault("stats", {})
    stats["trades_total"] = int(stats.get("trades_total", 0)) + 1

    if realized > 0:
        stats["trades_win"] = int(stats.get("trades_win", 0)) + 1
    elif realized < 0:
        stats["trades_loss"] = int(stats.get("trades_loss", 0)) + 1

    stats["largest_win"] = float(max(float(stats.get("largest_win", 0.0)), realized))
    stats["largest_loss"] = float(min(float(stats.get("largest_loss", 0.0)), realized))

    stats["pnl_total"] = float(state.get("pnl_eur", 0.0))
    tw = int(stats.get("trades_win", 0))
    tt = int(stats.get("trades_total", 0))
    stats["winrate"] = float(tw) / tt * 100.0 if tt > 0 else 0.0


# =========================================================
# Fees helpers
# =========================================================


def fee_in_quote(order: dict, quote: str) -> float:
    total = 0.0
    fees = []
    if "fees" in order and order["fees"]:
        fees.extend(order["fees"])
    if "fee" in order and order["fee"]:
        fees.append(order["fee"])

    for f in fees:
        cur = f.get("currency")
        cost = f.get("cost", 0) or 0
        try:
            cost = float(cost)
        except Exception:
            cost = 0.0
        if cur is None or cur == quote:
            total += cost

    return total


# =========================================================
# Bitvavo via ccxt
# =========================================================


def bv_client() -> ccxt.bitvavo:
    load_dotenv(ROOT / ".env")
    key = os.getenv("BITVAVO_API_KEY", "")
    sec = os.getenv("BITVAVO_API_SECRET", "")
    operator = os.getenv("BITVAVO_OPERATOR_ID", "")

    if not key or not sec:
        raise RuntimeError("BITVAVO_API_KEY en/of BITVAVO_API_SECRET ontbreken")
    if not operator:
        raise RuntimeError("BITVAVO_OPERATOR_ID ontbreekt (vereist door Bitvavo/ccxt)")

    exchange = ccxt.bitvavo(
        {
            "apiKey": key,
            "secret": sec,
            "enableRateLimit": True,
            "options": {"operatorId": operator},
        }
    )
    return exchange


def get_free_eur(exchange: ccxt.bitvavo) -> float:
    bal = exchange.fetch_balance()
    eur = bal.get("EUR") or {}
    return float(eur.get("free", 0.0) or 0.0)


def last_price(exchange: ccxt.bitvavo, market: str) -> float:
    t = exchange.fetch_ticker(market)
    return float(t["last"])


def orderbook_spread_pct(exchange: ccxt.bitvavo, market: str) -> float:
    ob = exchange.fetch_order_book(market, limit=1)
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    bid = float(bids[0][0]) if bids else 0.0
    ask = float(asks[0][0]) if asks else 0.0
    if bid and ask:
        return (ask - bid) / ((ask + bid) / 2) * 100.0
    return 999.0


def candles_df(exchange: ccxt.bitvavo, market: str, interval: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(market, timeframe=interval, limit=limit)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("UTC")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def place_market_buy(exchange: ccxt.bitvavo, market: str, amount_quote_eur: float):
    base, quote = market.split("/")
    price = last_price(exchange, market)
    amount_base = amount_quote_eur / price

    order = exchange.create_order(market, "market", "buy", amount_base)

    cost = float(order.get("cost", amount_quote_eur) or amount_quote_eur)
    fee_q = fee_in_quote(order, quote)
    cost_total = cost + fee_q

    filled = float(order.get("amount", amount_base) or amount_base)
    avg_price = cost_total / filled if filled else price

    return order, filled, cost_total, avg_price, fee_q


def place_market_sell_amount(exchange: ccxt.bitvavo, market: str, amount_base: float, last_px: float):
    base, quote = market.split("/")
    order = exchange.create_order(market, "market", "sell", amount_base)

    gross = float(order.get("cost", amount_base * last_px) or (amount_base * last_px))
    fee_q = fee_in_quote(order, quote)
    proceeds_net = gross - fee_q

    return order, proceeds_net, fee_q


# =========================================================
# Signalen (uptrend + downtrend)
# =========================================================


def compute_signals(df: pd.DataFrame, cfg: dict):
    sig = cfg["signals"]

    if sig.get("use_sma", True):
        df["sma_fast"] = sma(df["close"], sig["sma_fast"])
        df["sma_slow"] = sma(df["close"], sig["sma_slow"])
    if sig.get("use_rsi", True):
        df["rsi"] = rsi(df["close"], sig["rsi_len"])

    df = df.dropna().copy()
    if len(df) < 3:
        return df, False, False, "none"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    in_downtrend = False
    if sig.get("use_sma", True):
        in_downtrend = last["sma_fast"] < last["sma_slow"]

    buy = False
    sell = False
    mode = "up"

    # UP: trend-follow
    if not in_downtrend:
        mode = "up"
        buy = True
        sell = False

        if sig.get("use_sma", True):
            buy &= last["sma_fast"] > last["sma_slow"]
            sell |= last["sma_fast"] < last["sma_slow"]

        if sig.get("use_rsi", True):
            buy &= last["rsi"] >= sig.get("rsi_buy_min", 50)
            sell |= last["rsi"] <= sig.get("rsi_sell_max", 45)

    # DOWN: mean-reversion
    else:
        mode = "down"
        if sig.get("enable_downtrend", True) and sig.get("use_rsi", True):
            buy = last["rsi"] <= sig.get("down_rsi_buy_max", 30)

            if sig.get("down_require_rsi_uptick", True):
                buy &= last["rsi"] > prev["rsi"]

            sell = last["rsi"] >= sig.get("down_rsi_sell_min", 45)
        else:
            buy = False
            sell = False

    return df, bool(buy), bool(sell), mode


# =========================================================
# Risk / sizing
# =========================================================


def can_buy(base: str, state: dict, cfg: dict, free_eur: float):
    risk = cfg["risk"]
    eur_reserve = float(risk.get("eur_reserve", 0.0))

    usable_eur = max(0.0, free_eur - eur_reserve)
    if usable_eur <= 0:
        return False, "reserve_reached", 0.0

    open_cnt = sum(1 for p in state["positions"].values() if p.get("open", False))
    if open_cnt >= risk["max_open_positions"]:
        return False, "max_open_positions", 0.0

    cd = state["cooldown"].get(base)
    if cd:
        until = datetime.fromisoformat(cd)
        if now_ts() < until:
            return False, "cooldown", 0.0

    pos = state["positions"].get(base)
    if pos and pos.get("open", False) and risk.get("only_buy_if_not_in_position", True):
        return False, "already_in_position", 0.0

    per_coin_cap = usable_eur * (risk["max_pct_per_coin"] / 100.0)
    return True, "ok", per_coin_cap


def calc_stake(free_eur: float, cfg: dict, mode: str) -> float:
    risk = cfg["risk"]
    eur_reserve = float(risk.get("eur_reserve", 0.0))
    usable_eur = max(0.0, free_eur - eur_reserve)
    if usable_eur <= 0:
        return 0.0

    if risk.get("fixed_stake_quote", 0) > 0:
        stake = min(float(risk["fixed_stake_quote"]), usable_eur)
    else:
        stake = max(10.0, usable_eur * float(risk.get("stake_fraction", 0.0)))

    # Downtrend inzet verlagen (veiligheid)
    if mode == "down":
        mult = float(risk.get("down_stake_multiplier", 0.5))
        stake *= mult

    return stake


# =========================================================
# TP / SL (met downtrend overrides)
# =========================================================


def tp_sl_check(cur_price: float, pos: dict, prot: dict, mode: str):
    if mode == "down":
        tp_pct = float(prot.get("down_take_profit_pct", prot["take_profit_pct"]))
        sl_pct = float(prot.get("down_stop_loss_pct", prot["stop_loss_pct"]))
        tr_pct = float(prot.get("down_trailing_tp_pct", prot.get("trailing_tp_pct", 0.0)))
    else:
        tp_pct = float(prot["take_profit_pct"])
        sl_pct = float(prot["stop_loss_pct"])
        tr_pct = float(prot.get("trailing_tp_pct", 0.0))

    avg = pos["avg_price"]
    tp_hit = pct(cur_price, avg) >= tp_pct
    sl_hit = pct(cur_price, avg) <= -abs(sl_pct)

    if tr_pct > 0 and pos.get("peak", 0) > 0:
        giveback_hit = pct(cur_price, pos["peak"]) <= -abs(tr_pct)
        tp_hit = tp_hit or giveback_hit

    return tp_hit, sl_hit, tp_pct, sl_pct, tr_pct


# =========================================================
# Main
# =========================================================


def main():
    cfg = load_cfg()

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    exchange = bv_client()
    state = load_state()

    quote = cfg["quote"]
    interval = cfg["timeframe"]
    sleep_s = cfg["logging"]["loop_sleep_seconds"]
    limit = cfg["logging"]["candles_limit"]

    logging.info("Diamond-Pigs-stijl bot gestart (Bitvavo, ccxt, netto fees, extra logging)")

    while True:
        try:
            free_eur = get_free_eur(exchange)
            logging.info(f"Vrij {quote}: {free_eur:.2f}")

            for base in cfg["symbols"]:
                market = f"{base}/{quote}"

                try:
                    spread = orderbook_spread_pct(exchange, market)
                    if spread > cfg["risk"]["max_spread_pct"]:
                        logging.debug(f"{market} spread {spread:.3f}% > limiet")
                        continue

                    df = candles_df(exchange, market, interval, limit)
                    df, buy_sig, sell_sig, mode = compute_signals(df, cfg)
                    if df.empty or mode == "none":
                        continue

                    last_px = float(df.iloc[-1]["close"])

                    pos = state["positions"].get(base, {"open": False})

                    # peak update
                    if pos.get("open"):
                        pos["peak"] = max(pos.get("peak", 0.0), last_px)
                        state["positions"][base] = pos

                    # =========================
                    # VERKOOP
                    # =========================
                    if pos.get("open"):
                        tp_hit, sl_hit, tp_pct, sl_pct, tr_pct = tp_sl_check(
                            last_px, pos, cfg["protection"], mode
                        )
                        should_sell = sell_sig or tp_hit or sl_hit

                        if should_sell:
                            bot_amount = float(pos.get("amount_base", 0.0))

                            bal = exchange.fetch_balance()
                            asset = bal.get(base) or {}
                            avail = float(asset.get("free", 0.0) or 0.0)
                            amount_to_sell = min(bot_amount, avail)

                            if amount_to_sell > 0:
                                _, proceeds_net, fee_sell = place_market_sell_amount(
                                    exchange, market, amount_to_sell, last_px
                                )

                                cost_eur = float(pos.get("cost_eur", 0.0))
                                realized = proceeds_net - cost_eur
                                state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + realized
                                update_stats_after_trade(state, realized)
                                stats = state.get("stats", {})

                                holding_min = 0.0
                                opened_at = pos.get("opened_at")
                                if opened_at:
                                    try:
                                        dt_open = datetime.fromisoformat(opened_at)
                                        holding_min = (now_ts() - dt_open).total_seconds() / 60.0
                                    except Exception:
                                        holding_min = 0.0

                                entry_spread = pos.get("entry_spread_pct")

                                append_trade(
                                    {
                                        "ts": now_ts().isoformat(),
                                        "market": market,
                                        "mode": mode,
                                        "side": "SELL",
                                        "price": last_px,
                                        "amount_base": amount_to_sell,
                                        "proceeds_eur": proceeds_net,
                                        "avg_entry": pos["avg_price"],
                                        "realized_pnl_eur": realized,
                                        "fee_eur": fee_sell,
                                        "spread_pct": entry_spread,
                                        "holding_time_min": holding_min,
                                        "reason": "TP" if tp_hit else "SL" if sl_hit else "Signal",
                                        "tp_pct": tp_pct,
                                        "sl_pct": sl_pct,
                                        "trailing_tp_pct": tr_pct,
                                    }
                                )

                                # sluit positie + cooldown
                                state["positions"][base] = {"open": False}
                                mins = cfg["risk"]["cooldown_minutes"]
                                state["cooldown"][base] = (now_ts() + timedelta(minutes=mins)).isoformat()
                                save_state(state)

                                logging.info(
                                    f"{market} verkocht ({mode}). Netto PnL €{realized:.2f}, "
                                    f"totaal €{state['pnl_eur']:.2f}, "
                                    f"trades {stats.get('trades_total', 0)}, "
                                    f"winrate {stats.get('winrate', 0.0):.1f}%, "
                                    f"hold {holding_min:.1f} min"
                                )
                            else:
                                logging.warning(f"{market} geen eigen {base}-positie om te verkopen")

                    # =========================
                    # KOOP
                    # =========================
                    free_eur = get_free_eur(exchange)  # refresh na eventuele verkoop
                    can, reason, per_coin_cap = can_buy(base, state, cfg, free_eur)

                    if can and buy_sig and not pos.get("open", False):
                        stake = calc_stake(free_eur, cfg, mode)
                        stake = min(stake, float(per_coin_cap))

                        if stake >= 5.0:
                            _, filled_base, cost_total, avg_price, fee_buy = place_market_buy(
                                exchange, market, stake
                            )

                            state["positions"][base] = {
                                "open": True,
                                "mode": mode,
                                "avg_price": avg_price,
                                "cost_eur": cost_total,
                                "opened_at": now_ts().isoformat(),
                                "peak": avg_price,
                                "amount_base": filled_base,
                                "entry_spread_pct": spread,
                                "fee_buy_eur": fee_buy,
                            }

                            append_trade(
                                {
                                    "ts": now_ts().isoformat(),
                                    "market": market,
                                    "mode": mode,
                                    "side": "BUY",
                                    "price": avg_price,
                                    "amount_base": filled_base,
                                    "cost_eur": cost_total,
                                    "fee_eur": fee_buy,
                                    "spread_pct": spread,
                                    "holding_time_min": 0.0,
                                    "reason": "Signal",
                                }
                            )

                            save_state(state)
                            logging.info(
                                f"{market} gekocht ({mode}) voor netto €{cost_total:.2f} @ {avg_price:.4f}, "
                                f"spread {spread:.3f}%"
                            )
                        else:
                            logging.debug(f"{market} inzet te laag: {stake:.2f} ({mode})")
                    elif not can and reason == "reserve_reached":
                        logging.debug(f"{market} geen koop: EUR-reserve bereikt, free={free_eur:.2f}")

                except Exception as e:
                    logging.exception(f"Fout bij market {market}: {e}")

        except Exception as e:
            logging.exception(f"Hoofdlus fout: {e}")

        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
