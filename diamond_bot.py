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

# ---------- Indicatoren ----------

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

# ---------- Paden / helpers ----------

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "state.json"
TRADES_CSV = ROOT / "trades.csv"

def now_ts() -> datetime:
    return datetime.now(timezone.utc).astimezone()

def load_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "cooldown": {}, "pnl_eur": 0.0}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

def append_trade(row: dict) -> None:
    hdr = not TRADES_CSV.exists()
    df = pd.DataFrame([row])
    df.to_csv(TRADES_CSV, index=False, mode="a", header=hdr)

def pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0 if b else 0.0

# ---------- Fees helpers ----------

def fee_in_quote(order: dict, quote: str) -> float:
    """Som van alle fees in quote-valuta (EUR) uit een ccxt-order."""
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

# ---------- Bitvavo via ccxt ----------

def bv_client() -> ccxt.bitvavo:
    load_dotenv(ROOT / ".env")
    key = os.getenv("BITVAVO_API_KEY", "")
    sec = os.getenv("BITVAVO_API_SECRET", "")
    operator = os.getenv("BITVAVO_OPERATOR_ID", "")

    if not key or not sec:
        raise RuntimeError("BITVAVO_API_KEY en BITVAVO_API_SECRET ontbreken")
    if not operator:
        raise RuntimeError("BITVAVO_OPERATOR_ID ontbreekt (vereist door Bitvavo via ccxt)")

    exchange = ccxt.bitvavo({
        "apiKey": key,
        "secret": sec,
        "enableRateLimit": True,
        "options": {
            "operatorId": operator,
        },
    })
    return exchange

def get_free_eur(exchange: ccxt.bitvavo) -> float:
    bal = exchange.fetch_balance()
    eur = bal.get("EUR") or {}
    return float(eur.get("free", 0.0) or 0.0)

def price_ticker(exchange: ccxt.bitvavo, market: str) -> float:
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
    price = price_ticker(exchange, market)
    amount_base = amount_quote_eur / price

    order = exchange.create_order(market, "market", "buy", amount_base)

    cost = float(order.get("cost", amount_quote_eur) or amount_quote_eur)
    fee_q = fee_in_quote(order, quote)
    cost_total = cost + fee_q

    filled = float(order.get("amount", amount_base) or amount_base)
    avg_price = cost_total / filled if filled else price

    return order, filled, cost_total, avg_price, fee_q

def place_market_sell_amount(
    exchange: ccxt.bitvavo, market: str, amount_base: float, last_price: float
):
    base, quote = market.split("/")
    order = exchange.create_order(market, "market", "sell", amount_base)

    gross = float(order.get("cost", amount_base * last_price) or (amount_base * last_price))
    fee_q = fee_in_quote(order, quote)
    proceeds_net = gross - fee_q

    return order, proceeds_net, fee_q

# ---------- Signalen ----------

def compute_signals(df: pd.DataFrame, cfg: dict):
    if cfg["signals"]["use_sma"]:
        df["sma_fast"] = sma(df["close"], cfg["signals"]["sma_fast"])
        df["sma_slow"] = sma(df["close"], cfg["signals"]["sma_slow"])
    if cfg["signals"]["use_rsi"]:
        df["rsi"] = rsi(df["close"], cfg["signals"]["rsi_len"])
    df = df.dropna().copy()
    if df.empty:
        return df, False, False

    last = df.iloc[-1]
    buy = True
    sell = False

    if cfg["signals"]["use_sma"]:
        buy &= last["sma_fast"] > last["sma_slow"]
        sell |= last["sma_fast"] < last["sma_slow"]

    if cfg["signals"]["use_rsi"]:
        buy &= last["rsi"] >= cfg["signals"]["rsi_buy_min"]
        sell |= last["rsi"] <= cfg["signals"]["rsi_sell_max"]

    return df, bool(buy), bool(sell)

# ---------- Risk / sizing ----------

def can_buy(base: str, state: dict, cfg: dict, free_eur: float):
    risk = cfg["risk"]
    open_cnt = sum(1 for p in state["positions"].values() if p.get("open", False))
    if open_cnt >= risk["max_open_positions"]:
        return False, "max_open_positions"

    cd = state["cooldown"].get(base)
    if cd:
        until = datetime.fromisoformat(cd)
        if now_ts() < until:
            return False, "cooldown"

    pos = state["positions"].get(base)
    if pos and pos.get("open", False) and risk["only_buy_if_not_in_position"]:
        return False, "already_in_position"

    per_coin_cap = free_eur * (risk["max_pct_per_coin"] / 100.0)
    return True, per_coin_cap

def calc_stake(free_eur: float, cfg: dict) -> float:
    risk = cfg["risk"]
    if risk["fixed_stake_quote"] > 0:
        return min(risk["fixed_stake_quote"], free_eur)
    return max(10.0, free_eur * risk["stake_fraction"])

# ---------- TP / SL ----------

def tp_sl_check(cur_price: float, pos: dict, prot: dict):
    avg = pos["avg_price"]
    tp_hit = pct(cur_price, avg) >= prot["take_profit_pct"]
    sl_hit = pct(cur_price, avg) <= -abs(prot["stop_loss_pct"])

    if prot.get("trailing_tp_pct", 0) > 0 and pos.get("peak", 0) > 0:
        trail_pct = prot["trailing_tp_pct"]
        giveback_hit = pct(cur_price, pos["peak"]) <= -abs(trail_pct)
        tp_hit = tp_hit or giveback_hit

    return tp_hit, sl_hit

# ---------- Main loop ----------

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

    logging.info("Diamond-Pigs-stijl bot (ccxt, netto fees, extra logging) gestart")

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
                    df, buy_sig, sell_sig = compute_signals(df, cfg)
                    if df.empty:
                        continue
                    last_price = float(df.iloc[-1]["close"])

                    pos = state["positions"].get(base, {"open": False})

                    if pos.get("open"):
                        pos["peak"] = max(pos.get("peak", 0.0), last_price)
                        state["positions"][base] = pos

                    # ---------- Verkoop ----------
                    if pos.get("open"):
                        tp_hit, sl_hit = tp_sl_check(last_price, pos, cfg["protection"])
                        should_sell = sell_sig or tp_hit or sl_hit
                        if should_sell:
                            bot_amount = float(pos.get("amount_base", 0.0))

                            bal = exchange.fetch_balance()
                            asset = bal.get(base) or {}
                            avail = float(asset.get("free", 0.0) or 0.0)

                            amount_to_sell = min(bot_amount, avail)

                            if amount_to_sell > 0:
                                _, proceeds_net, fee_sell_eur = place_market_sell_amount(
                                    exchange, market, amount_to_sell, last_price
                                )
                                cost_eur = float(pos.get("cost_eur", 0.0))
                                realized = proceeds_net - cost_eur
                                state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + realized

                                holding_min = 0.0
                                opened_at = pos.get("opened_at")
                                if opened_at:
                                    try:
                                        dt_open = datetime.fromisoformat(opened_at)
                                        holding_min = (now_ts() - dt_open).total_seconds() / 60.0
                                    except Exception:
                                        holding_min = 0.0

                                entry_spread = pos.get("entry_spread_pct")

                                append_trade({
                                    "ts": now_ts().isoformat(),
                                    "market": market,
                                    "side": "SELL",
                                    "price": last_price,
                                    "amount_base": amount_to_sell,
                                    "proceeds_eur": proceeds_net,
                                    "avg_entry": pos["avg_price"],
                                    "realized_pnl_eur": realized,
                                    "fee_eur": fee_sell_eur,
                                    "spread_pct": entry_spread,
                                    "holding_time_min": holding_min,
                                    "reason": "TP" if tp_hit else "SL" if sl_hit else "Signal",
                                })

                                state["positions"][base] = {"open": False}
                                mins = cfg["risk"]["cooldown_minutes"]
                                state["cooldown"][base] = (
                                    now_ts() + timedelta(minutes=mins)
                                ).isoformat()
                                save_state(state)
                                logging.info(
                                    f"{market} verkocht. Netto PnL €{realized:.2f}, hold {holding_min:.1f} min"
                                )
                            else:
                                logging.warning(f"{market} geen eigen {base}-positie om te verkopen")

                    # ---------- Koop ----------
                    can, cap = can_buy(base, state, cfg, free_eur)
                    if can and buy_sig:
                        stake = calc_stake(free_eur, cfg)
                        if isinstance(cap, (int, float)):
                            stake = min(stake, cap)
                        if stake >= 5.0:
                            order, filled_base, cost_total_eur, avg_price, fee_buy_eur = place_market_buy(
                                exchange, market, stake
                            )

                            state["positions"][base] = {
                                "open": True,
                                "avg_price": avg_price,
                                "cost_eur": cost_total_eur,
                                "opened_at": now_ts().isoformat(),
                                "peak": avg_price,
                                "amount_base": filled_base,
                                "entry_spread_pct": spread,
                                "fee_buy_eur": fee_buy_eur,
                            }

                            append_trade({
                                "ts": now_ts().isoformat(),
                                "market": market,
                                "side": "BUY",
                                "price": avg_price,
                                "amount_base": filled_base,
                                "cost_eur": cost_total_eur,
                                "fee_eur": fee_buy_eur,
                                "spread_pct": spread,
                                "holding_time_min": 0.0,
                                "reason": "Signal",
                            })

                            save_state(state)
                            logging.info(
                                f"{market} gekocht voor netto €{cost_total_eur:.2f} @ {avg_price:.2f}, spread {spread:.3f}%"
                            )
                        else:
                            logging.debug(f"{market} inzet te laag: {stake:.2f}")

                except Exception as e:
                    logging.exception(f"Fout bij market {market}: {e}")

        except Exception as e:
            logging.exception(f"Hoofdlus fout: {e}")

        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
