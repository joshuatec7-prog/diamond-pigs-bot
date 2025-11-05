import os, time, json, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from bitvavo import Bitvavo

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

def now_ts():
    return datetime.now(timezone.utc).astimezone()

def load_cfg():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "cooldown": {}, "pnl_eur": 0.0}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def append_trade(row: dict):
    hdr = not TRADES_CSV.exists()
    df = pd.DataFrame([row])
    df.to_csv(TRADES_CSV, index=False, mode="a", header=hdr)

def pct(a, b):
    return (a - b) / b * 100.0 if b else 0.0

# ---------- Bitvavo ----------
def bv_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("BITVAVO_API_KEY", "")
    sec = os.getenv("BITVAVO_API_SECRET", "")
    if not key or not sec:
        raise RuntimeError("BITVAVO_API_KEY en BITVAVO_API_SECRET ontbreken")
    return Bitvavo({
        'APIKEY': key,
        'APISECRET': sec,
        'RESTURL': 'https://api.bitvavo.com/v2/',
        'WSURL': 'wss://ws.bitvavo.com/v2/'
    })

def get_free_eur(bv):
    bal = bv.balance({'symbol': 'EUR'})
    if isinstance(bal, list) and bal:
        return float(bal[0].get('available', '0') or 0)
    return 0.0

def price_ticker(bv, market):
    t = bv.tickerPrice({'market': market})
    return float(t[0]['price'])

def orderbook_spread_pct(bv, market):
    ob = bv.getBook({'market': market, 'depth': 1})
    bid = float(ob.get('bids', [[0, 0]])[0][0]) if ob.get('bids') else 0.0
    ask = float(ob.get('asks', [[0, 0]])[0][0]) if ob.get('asks') else 0.0
    if bid and ask:
        return (ask - bid) / ((ask + bid) / 2) * 100.0
    return 999.0

def candles_df(bv, market, interval, limit):
    raw = bv.candles(market, interval, {'limit': limit})
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert('UTC')
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)

def place_market_buy(bv, market, amount_quote_eur):
    params = {"amountQuote": f"{amount_quote_eur:.2f}"}
    return bv.placeOrder(market, "buy", "market", params)

def place_market_sell_amount(bv, market, amount_base):
    params = {"amount": f"{amount_base:.8f}"}
    return bv.placeOrder(market, "sell", "market", params)

# ---------- Signalen ----------
def compute_signals(df, cfg):
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
def can_buy(base, state, cfg, free_eur):
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

def calc_stake(free_eur, cfg):
    risk = cfg["risk"]
    if risk["fixed_stake_quote"] > 0:
        return min(risk["fixed_stake_quote"], free_eur)
    return max(10.0, free_eur * risk["stake_fraction"])

# ---------- TP / SL ----------
def tp_sl_check(cur_price, pos, prot):
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

    bv = bv_client()
    state = load_state()

    quote = cfg["quote"]
    interval = cfg["timeframe"]
    sleep_s = cfg["logging"]["loop_sleep_seconds"]
    limit = cfg["logging"]["candles_limit"]

    logging.info("Diamond-Pigs-stijl bot gestart")

    while True:
        try:
            free_eur = get_free_eur(bv)
            logging.info(f"Vrij {quote}: {free_eur:.2f}")

            for base in cfg["symbols"]:
                market = f"{base}-{quote}"
                try:
                    spread = orderbook_spread_pct(bv, market)
                    if spread > cfg["risk"]["max_spread_pct"]:
                        logging.debug(f"{market} spread {spread:.3f}% > limiet")
                        continue

                    df = candles_df(bv, market, interval, limit)
                    df, buy_sig, sell_sig = compute_signals(df, cfg)
                    if df.empty:
                        continue
                    last_price = float(df.iloc[-1]["close"])

                    pos = state["positions"].get(base, {"open": False})
                    if pos.get("open"):
                        pos["peak"] = max(pos.get("peak", 0.0), last_price)

                    # ---------- Verkoop ----------
                    if pos.get("open"):
                        tp_hit, sl_hit = tp_sl_check(last_price, pos, cfg["protection"])
                        should_sell = sell_sig or tp_hit or sl_hit
                        if should_sell:
                            bot_amount = float(pos.get("amount_base", 0.0))
                            bal = bv.balance({'symbol': base})
                            avail = 0.0
                            if isinstance(bal, list) and bal:
                                avail = float(bal[0].get('available', '0') or 0.0)
                            amount_to_sell = min(bot_amount, avail)
                            if amount_to_sell > 0:
                                place_market_sell_amount(bv, market, amount_to_sell)
                                proceeds = amount_to_sell * last_price
                                cost = float(pos.get("cost_eur", 0.0))
                                realized = proceeds - cost
                                state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + realized
                                append_trade({
                                    "ts": now_ts().isoformat(),
                                    "market": market,
                                    "side": "SELL",
                                    "price": last_price,
                                    "amount_base": amount_to_sell,
                                    "proceeds_eur": proceeds,
                                    "avg_entry": pos["avg_price"],
                                    "realized_pnl_eur": realized,
                                    "reason": "TP" if tp_hit else "SL" if sl_hit else "Signal",
                                })
                                state["positions"][base] = {"open": False}
                                mins = cfg["risk"]["cooldown_minutes"]
                                state["cooldown"][base] = (now_ts() + timedelta(minutes=mins)).isoformat()
                                save_state(state)
                                logging.info(f"{market} verkocht. PnL €{realized:.2f}")
                            else:
                                logging.warning(f"{market} geen eigen {base}-positie om te verkopen")

                    # ---------- Koop ----------
                    can, cap = can_buy(base, state, cfg, free_eur)
                    if can and buy_sig:
                        stake = calc_stake(free_eur, cfg)
                        if isinstance(cap, (int, float)):
                            stake = min(stake, cap)
                        if stake >= 5.0:
                            order = place_market_buy(bv, market, stake)
                            filled_eur, filled_base = 0.0, 0.0
                            if isinstance(order, dict):
                                for f in order.get("fills", []):
                                    filled_eur += float(f.get("amountQuote", 0) or 0)
                                    filled_base += float(f.get("amount", 0) or 0)
                            if filled_eur == 0 or filled_base == 0:
                                price = price_ticker(bv, market)
                                filled_eur = stake
                                filled_base = stake / price
                            avg_price = filled_eur / filled_base if filled_base else last_price
                            state["positions"][base] = {
                                "open": True,
                                "avg_price": avg_price,
                                "cost_eur": filled_eur,
                                "opened_at": now_ts().isoformat(),
                                "peak": avg_price,
                                "amount_base": filled_base,
                            }
                            append_trade({
                                "ts": now_ts().isoformat(),
                                "market": market,
                                "side": "BUY",
                                "price": avg_price,
                                "amount_base": filled_base,
                                "cost_eur": filled_eur,
                                "reason": "Signal",
                            })
                            save_state(state)
                            logging.info(f"{market} gekocht voor €{filled_eur:.2f} @ {avg_price:.2f}")
                        else:
                            logging.debug(f"{market} inzet te laag: {stake:.2f}")
                except Exception as e:
                    logging.exception(f"Fout bij {market}: {e}")
        except Exception as e:
            logging.exception(f"Hoofdlus fout: {e}")
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
