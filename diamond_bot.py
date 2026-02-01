#!/usr/bin/env python3
# Diamond-Pigs style bot (Trend v2) - Bitvavo via ccxt (ENGLISH config)
#
# Goals:
# - Fully ENGLISH config keys (quote/symbols/timeframe/signals/risk/fees/logging)
# - Accepts booleans: WAAR/waar/true/True/yes/ja/1/on/aan
# - Accepts comma decimals: "0,12" and dot decimals: "0.12"
# - NEVER sells coins it did not buy itself:
#     -> Only sells positions recorded in state.json (opened_by_bot=true)
# - Logs net PnL including estimated fees per SELL + cumulative in state.json
# - Safe balance handling (retries/backoff)
#
# Requirements: ccxt, pandas, pyyaml, python-dotenv
# Env:
#   BITVAVO_API_KEY, BITVAVO_API_SECRET
#   (recommended) BITVAVO_OPERATOR_ID  (some ccxt builds require this)
# Optional:
#   CFG_FILE=/path/to/config.yaml
#   STATE_FILE=/path/to/state.json

import os
import time
import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
import ccxt

LOG = logging.getLogger("diamond")


# ------------------ Helpers: parsing ------------------

def to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        return s in ("true", "waar", "yes", "ja", "1", "on", "aan", "y")
    return default


def to_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%", "")
        s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return default
    return default


def to_int(v, default=0):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        try:
            return int(float(s))
        except ValueError:
            return default
    return default


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ------------------ Indicators ------------------

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


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


# ------------------ Files & logging ------------------

ROOT = Path(__file__).parent
CFG_FILE = Path(os.getenv("CFG_FILE", str(ROOT / "config.yaml")))
STATE_FILE = Path(os.getenv("STATE_FILE", str(ROOT / "state.json")))
TX_CSV = ROOT / "transactions.csv"


def setup_logging(level: str):
    level = (level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------ Config ------------------

ALLOWED_TOP_KEYS = {"quote", "symbols", "timeframe", "signals", "risk", "fees", "logging"}
ALLOWED_SIGNALS_KEYS = {
    "use_sma", "sma_fast", "sma_slow",
    "use_rsi", "rsi_len", "rsi_buy_min",
    "use_atr", "atr_len", "atr_tp_mult", "atr_sl_mult",
    "exit_on_trend_break",
    "use_atr_filter", "min_atr_pct",
}
ALLOWED_RISK_KEYS = {
    "fixed_stake_quote", "max_open_positions", "only_buy_if_not_in_position",
    "cooldown_minutes", "max_spread_pct", "eur_reserve",
}
ALLOWED_FEES_KEYS = {"taker_fee_pct"}
ALLOWED_LOG_KEYS = {"level", "loop_sleep_seconds", "candles_limit"}


def _warn_unknown_keys(section_name: str, data: dict, allowed: set):
    if not isinstance(data, dict):
        return
    unknown = sorted([k for k in data.keys() if k not in allowed])
    if unknown:
        LOG.warning(f"Config: unknown keys in '{section_name}': {unknown} (ignored)")


def load_cfg() -> dict:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Config not found: {CFG_FILE}")

    raw = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping (top-level dict).")

    _warn_unknown_keys("root", raw, ALLOWED_TOP_KEYS)

    cfg = dict(raw)
    cfg.setdefault("quote", "EUR")
    cfg.setdefault("symbols", ["BTC", "ETH"])
    cfg.setdefault("timeframe", "15m")
    cfg.setdefault("signals", {})
    cfg.setdefault("risk", {})
    cfg.setdefault("fees", {})
    cfg.setdefault("logging", {})

    sig = cfg["signals"]
    rk = cfg["risk"]
    fees = cfg["fees"]
    lg = cfg["logging"]

    _warn_unknown_keys("signals", sig, ALLOWED_SIGNALS_KEYS)
    _warn_unknown_keys("risk", rk, ALLOWED_RISK_KEYS)
    _warn_unknown_keys("fees", fees, ALLOWED_FEES_KEYS)
    _warn_unknown_keys("logging", lg, ALLOWED_LOG_KEYS)

    # defaults: signals
    sig.setdefault("use_sma", True)
    sig.setdefault("sma_fast", 20)
    sig.setdefault("sma_slow", 60)

    sig.setdefault("use_rsi", True)
    sig.setdefault("rsi_len", 14)
    sig.setdefault("rsi_buy_min", 58)

    sig.setdefault("use_atr", True)
    sig.setdefault("atr_len", 14)
    sig.setdefault("atr_tp_mult", 2.0)
    sig.setdefault("atr_sl_mult", 1.8)

    sig.setdefault("exit_on_trend_break", True)

    # optional filter: only trade if ATR% >= min_atr_pct
    sig.setdefault("use_atr_filter", False)
    sig.setdefault("min_atr_pct", 0.25)

    # defaults: risk
    rk.setdefault("fixed_stake_quote", 15.0)
    rk.setdefault("max_open_positions", 2)
    rk.setdefault("only_buy_if_not_in_position", True)
    rk.setdefault("cooldown_minutes", 60)
    rk.setdefault("max_spread_pct", 0.12)   # NOTE: this is percent, e.g. 0.12 means 0.12%
    rk.setdefault("eur_reserve", 50.0)

    # defaults: fees
    fees.setdefault("taker_fee_pct", 0.25)

    # defaults: logging
    lg.setdefault("level", "INFO")
    lg.setdefault("loop_sleep_seconds", 30)
    lg.setdefault("candles_limit", 400)

    # coerce types
    cfg["quote"] = str(cfg["quote"]).upper()
    cfg["symbols"] = list(cfg["symbols"])
    cfg["timeframe"] = str(cfg["timeframe"])

    sig["use_sma"] = to_bool(sig.get("use_sma"), True)
    sig["sma_fast"] = to_int(sig.get("sma_fast"), 20)
    sig["sma_slow"] = to_int(sig.get("sma_slow"), 60)

    sig["use_rsi"] = to_bool(sig.get("use_rsi"), True)
    sig["rsi_len"] = to_int(sig.get("rsi_len"), 14)
    sig["rsi_buy_min"] = to_float(sig.get("rsi_buy_min"), 58)

    sig["use_atr"] = to_bool(sig.get("use_atr"), True)
    sig["atr_len"] = to_int(sig.get("atr_len"), 14)
    sig["atr_tp_mult"] = to_float(sig.get("atr_tp_mult"), 2.0)
    sig["atr_sl_mult"] = to_float(sig.get("atr_sl_mult"), 1.8)

    sig["exit_on_trend_break"] = to_bool(sig.get("exit_on_trend_break"), True)

    sig["use_atr_filter"] = to_bool(sig.get("use_atr_filter"), False)
    sig["min_atr_pct"] = to_float(sig.get("min_atr_pct"), 0.25)

    rk["fixed_stake_quote"] = to_float(rk.get("fixed_stake_quote"), 15.0)
    rk["max_open_positions"] = to_int(rk.get("max_open_positions"), 2)
    rk["only_buy_if_not_in_position"] = to_bool(rk.get("only_buy_if_not_in_position"), True)
    rk["cooldown_minutes"] = to_int(rk.get("cooldown_minutes"), 60)
    rk["max_spread_pct"] = to_float(rk.get("max_spread_pct"), 0.12)
    rk["eur_reserve"] = to_float(rk.get("eur_reserve"), 50.0)

    fees["taker_fee_pct"] = to_float(fees.get("taker_fee_pct"), 0.25)

    lg["level"] = str(lg.get("level", "INFO")).upper()
    lg["loop_sleep_seconds"] = to_int(lg.get("loop_sleep_seconds"), 30)
    lg["candles_limit"] = to_int(lg.get("candles_limit"), 400)

    cfg["_cfg_path"] = str(CFG_FILE)
    return cfg


# ------------------ State & CSV ------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "positions": {},     # market -> {opened_by_bot, base_amount, entry_price, entry_ts, entry_cost, entry_fee}
        "cooldown": {},      # market -> next_allowed_ts (unix)
        "pnl_quote": 0.0,    # cumulative net realized pnl in quote
        "trades": 0,
        "wins": 0,
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_csv_header():
    if TX_CSV.exists():
        return
    cols = [
        "ts", "market", "mode", "side", "price", "base_amount",
        "fees_quote", "spread_pct", "net_pnl_quote", "holding_time_min", "reason"
    ]
    pd.DataFrame([], columns=cols).to_csv(TX_CSV, index=False)


def append_tx(row: dict):
    ensure_csv_header()
    pd.DataFrame([row]).to_csv(TX_CSV, index=False, mode="a", header=False)


# ------------------ Exchange ------------------

def make_exchange():
    api_key = os.getenv("BITVAVO_API_KEY", "").strip()
    api_secret = os.getenv("BITVAVO_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Missing BITVAVO_API_KEY / BITVAVO_API_SECRET in environment variables.")

    ex = ccxt.bitvavo({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    opid = os.getenv("BITVAVO_OPERATOR_ID", "").strip()
    if opid:
        try:
            ex.options = ex.options or {}
            ex.options["operatorId"] = int(opid)
        except Exception:
            LOG.warning("BITVAVO_OPERATOR_ID must be an integer. Ignored.")

    return ex


def safe_fetch_balance(ex, retries=5, base_sleep=1.0):
    last_err = None
    for i in range(retries):
        try:
            return ex.fetch_balance()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** i)
            LOG.warning(f"Fetch balance failed ({i+1}/{retries}): {e}. Sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Fetch balance failed after {retries} attempts: {last_err}")


# ------------------ Trading logic ------------------

def market_symbol(base: str, quote: str) -> str:
    return f"{base}/{quote}"


def get_free_quote(balance: dict, quote: str) -> float:
    try:
        free = balance.get("free", {}).get(quote, 0.0)
        return float(free or 0.0)
    except Exception:
        return 0.0


def fetch_candles(ex, market: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) < 50:
        raise RuntimeError("Not enough candle data.")
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def fetch_ticker(ex, market: str) -> dict:
    return ex.fetch_ticker(market)


def calc_spread_pct(ticker: dict) -> float:
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    if not bid or not ask or bid <= 0:
        return 999.0
    return (ask - bid) / ask * 100.0


def fee_estimate_quote(amount_quote: float, taker_fee_pct: float) -> float:
    return amount_quote * (taker_fee_pct / 100.0)


def compute_signal(df: pd.DataFrame, cfg: dict) -> dict:
    sig = cfg["signals"]
    closes = df["close"]

    # SMA
    if sig["use_sma"]:
        df["sma_fast"] = sma(closes, sig["sma_fast"])
        df["sma_slow"] = sma(closes, sig["sma_slow"])
    else:
        df["sma_fast"] = float("nan")
        df["sma_slow"] = float("nan")

    # RSI
    if sig["use_rsi"]:
        df["rsi"] = rsi(closes, sig["rsi_len"])
    else:
        df["rsi"] = float("nan")

    # ATR
    if sig["use_atr"] or sig["use_atr_filter"]:
        df["atr"] = atr(df, sig["atr_len"])
    else:
        df["atr"] = 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_close = float(last["close"])

    # Trend
    trend_up = True
    cross_up = True
    trend_break = False
    if sig["use_sma"]:
        trend_up = (last["sma_fast"] > last["sma_slow"])
        cross_up = (prev["sma_fast"] <= prev["sma_slow"]) and (last["sma_fast"] > last["sma_slow"])
        trend_break = (last["sma_fast"] < last["sma_slow"])

    # RSI check
    rsi_val = float(last["rsi"]) if sig["use_rsi"] and not math.isnan(float(last["rsi"])) else 50.0
    rsi_ok = (rsi_val >= sig["rsi_buy_min"]) if sig["use_rsi"] else True

    # ATR filter
    atr_val = float(last["atr"]) if not math.isnan(float(last.get("atr", 0.0))) else 0.0
    atr_pct = (atr_val / last_close * 100.0) if last_close > 0 else 0.0
    atr_filter_ok = True
    if sig["use_atr_filter"]:
        atr_filter_ok = (atr_pct >= sig["min_atr_pct"])

    return {
        "trend_up": bool(trend_up),
        "cross_up": bool(cross_up),
        "trend_break": bool(trend_break),
        "rsi": float(rsi_val),
        "rsi_ok": bool(rsi_ok),
        "atr": float(atr_val),
        "atr_pct": float(atr_pct),
        "atr_filter_ok": bool(atr_filter_ok),
        "last_close": last_close,
    }


def should_buy(signal: dict, in_position: bool, cfg: dict) -> (bool, str):
    sig = cfg["signals"]

    if in_position:
        return False, "Already in position"

    if sig["use_sma"] and not signal["trend_up"]:
        return False, "No uptrend (SMA)"

    if sig["use_rsi"] and not signal["rsi_ok"]:
        return False, f"RSI {signal['rsi']:.1f} below threshold"

    if sig["use_atr_filter"] and not signal["atr_filter_ok"]:
        return False, f"ATR% {signal['atr_pct']:.3f} below min"

    # cross_up is quality, not required (keeps trades from becoming too rare)
    return True, "Signal"


def should_sell(signal: dict, pos: dict, cfg: dict, last_price: float) -> (bool, str, float, float):
    sig = cfg["signals"]
    entry = float(pos["entry_price"])

    atr_val = float(signal.get("atr", 0.0) or 0.0)
    if atr_val <= 0:
        # fallback if ATR missing: 0.2% of entry (approx)
        atr_val = max(entry * 0.002, 0.01)

    tp = entry + sig["atr_tp_mult"] * atr_val
    sl = entry - sig["atr_sl_mult"] * atr_val

    if last_price >= tp:
        return True, "ATR_TP", tp, sl
    if last_price <= sl:
        return True, "ATR_SL", tp, sl
    if sig["exit_on_trend_break"] and signal.get("trend_break", False):
        return True, "TREND_BREAK", tp, sl

    return False, "Hold", tp, sl


def amount_to_precision_safe(ex, market: str, amount: float) -> float:
    try:
        s = ex.amount_to_precision(market, amount)
        return float(s)
    except Exception:
        return float(amount)


def market_limits(ex, market: str):
    mk = ex.market(market)
    min_amount = float(mk.get("limits", {}).get("amount", {}).get("min", 0) or 0)
    min_cost = float(mk.get("limits", {}).get("cost", {}).get("min", 0) or 0)
    return min_amount, min_cost


def place_market_buy(ex, market: str, stake_quote: float, taker_fee_pct: float):
    ticker = fetch_ticker(ex, market)
    ask = float(ticker.get("ask") or 0.0)
    if ask <= 0:
        raise RuntimeError("No ask price.")

    # Convert quote stake to base amount (approx), then precision it
    base_amount = stake_quote / ask
    base_amount = amount_to_precision_safe(ex, market, base_amount)

    # Respect minima
    min_amount, min_cost = market_limits(ex, market)
    if min_cost and stake_quote < min_cost:
        raise RuntimeError(f"Stake {stake_quote:.2f} below min cost {min_cost:.2f}")
    if min_amount and base_amount < min_amount:
        raise RuntimeError(f"Amount {base_amount:.8f} below min amount {min_amount}")

    order = ex.create_order(market, "market", "buy", base_amount)

    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or ask)
    cost = float(order.get("cost") or (filled * avg))

    fee_quote = fee_estimate_quote(cost, taker_fee_pct)
    return order, filled, avg, cost, fee_quote, ticker


def place_market_sell(ex, market: str, base_amount: float, taker_fee_pct: float):
    ticker = fetch_ticker(ex, market)
    bid = float(ticker.get("bid") or 0.0)
    if bid <= 0:
        raise RuntimeError("No bid price.")

    base_amount = amount_to_precision_safe(ex, market, base_amount)
    order = ex.create_order(market, "market", "sell", base_amount)

    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or bid)
    proceeds = float(order.get("cost") or (filled * avg))  # quote received

    fee_quote = fee_estimate_quote(proceeds, taker_fee_pct)
    return order, filled, avg, proceeds, fee_quote, ticker


# ------------------ Main loop ------------------

def main():
    load_dotenv()

    cfg = load_cfg()
    setup_logging(cfg["logging"]["level"])

    LOG.info(f"Trend bot started. Config={cfg.get('_cfg_path')}")
    LOG.info(
        "Booleans | "
        f"use_sma={cfg['signals']['use_sma']} use_rsi={cfg['signals']['use_rsi']} "
        f"use_atr_filter={cfg['signals']['use_atr_filter']} "
        f"exit_on_trend_break={cfg['signals']['exit_on_trend_break']}"
    )

    ex = make_exchange()
    ex.load_markets()

    state = load_state()
    ensure_csv_header()

    quote = cfg["quote"]
    bases = cfg["symbols"]
    timeframe = cfg["timeframe"]
    candles_limit = int(cfg["logging"]["candles_limit"])

    stake = float(cfg["risk"]["fixed_stake_quote"])
    max_open = int(cfg["risk"]["max_open_positions"])
    only_buy_if_not_in_pos = bool(cfg["risk"]["only_buy_if_not_in_position"])
    cooldown_min = int(cfg["risk"]["cooldown_minutes"])
    max_spread_pct = float(cfg["risk"]["max_spread_pct"])
    eur_reserve = float(cfg["risk"]["eur_reserve"])

    taker_fee_pct = float(cfg["fees"]["taker_fee_pct"])
    sleep_s = int(cfg["logging"]["loop_sleep_seconds"])

    while True:
        try:
            balance = safe_fetch_balance(ex, retries=5, base_sleep=1.0)
            free_quote = get_free_quote(balance, quote)

            # For decisions we keep an internal "virtual" free_quote that we update after trades,
            # but periodically it's refreshed from balance (each outer loop).
            v_free_quote = float(free_quote)

            LOG.info(f"Free {quote}: {v_free_quote:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            # Compute open positions from state (bot-only)
            positions = state.get("positions", {}) or {}
            open_positions = len(positions)

            for base in bases:
                market = market_symbol(base, quote)

                # cooldown
                now = time.time()
                next_ok = float((state.get("cooldown", {}) or {}).get(market, 0) or 0)
                if now < next_ok:
                    continue

                in_pos = market in positions

                # Spread check
                try:
                    tkr = fetch_ticker(ex, market)
                    spr = calc_spread_pct(tkr)
                except Exception as e:
                    LOG.warning(f"{market}: ticker fetch failed: {e}")
                    continue

                if spr > max_spread_pct:
                    continue

                # Candles + signal
                try:
                    df = fetch_candles(ex, market, timeframe, candles_limit)
                    sig = compute_signal(df, cfg)
                except Exception as e:
                    LOG.warning(f"{market}: candle fetch failed: {e}")
                    continue

                last_price = float(sig["last_close"])

                # ---------------- SELL (ONLY if opened_by_bot) ----------------
                if in_pos:
                    pos = positions[market]

                    # Hard safety: do not sell if this position wasn't opened by this bot.
                    if not to_bool(pos.get("opened_by_bot", False), False):
                        continue

                    do_sell, reason, tp, sl = should_sell(sig, pos, cfg, last_price)
                    if do_sell:
                        base_amount = float(pos["base_amount"])
                        try:
                            _, filled, avg, proceeds, sell_fee, _ticker = place_market_sell(
                                ex, market, base_amount, taker_fee_pct
                            )
                        except Exception as e:
                            LOG.error(f"{market}: sell failed: {e}")
                            continue

                        entry_cost = float(pos.get("entry_cost_quote", 0.0))
                        entry_fee = float(pos.get("entry_fee_quote", 0.0))

                        net_pnl = (proceeds - entry_cost) - (entry_fee + sell_fee)

                        # holding time
                        hold_min = 0.0
                        entry_ts = pos.get("entry_ts")
                        if entry_ts:
                            try:
                                t0 = datetime.fromisoformat(entry_ts)
                                dt = datetime.now(timezone.utc).astimezone() - t0
                                hold_min = dt.total_seconds() / 60.0
                            except Exception:
                                hold_min = 0.0

                        state["pnl_quote"] = float(state.get("pnl_quote", 0.0)) + float(net_pnl)
                        state["trades"] = int(state.get("trades", 0)) + 1
                        if net_pnl > 0:
                            state["wins"] = int(state.get("wins", 0)) + 1

                        # cooldown
                        state.setdefault("cooldown", {})
                        state["cooldown"][market] = time.time() + cooldown_min * 60

                        # remove position
                        del state["positions"][market]
                        save_state(state)

                        # update internal counters
                        open_positions = max(0, open_positions - 1)
                        v_free_quote += float(proceeds)  # proceeds come back to quote (approx)

                        row = {
                            "ts": now_iso(),
                            "market": market,
                            "mode": "trend",
                            "side": "SELL",
                            "price": avg,
                            "base_amount": filled,
                            "fees_quote": (entry_fee + sell_fee),
                            "spread_pct": spr,
                            "net_pnl_quote": net_pnl,
                            "holding_time_min": hold_min,
                            "reason": reason,
                        }
                        append_tx(row)

                        winrate = (state["wins"] / state["trades"] * 100.0) if state["trades"] else 0.0
                        LOG.info(
                            f"{market} SOLD. Net PnL {quote} {net_pnl:,.2f}, total {quote} {state['pnl_quote']:,.2f}, "
                            f"trades {state['trades']}, winrate {winrate:.1f}%, hold {hold_min:.1f} min, reason {reason}"
                            .replace(",", "X").replace(".", ",").replace("X", ".")
                        )

                # ---------------- BUY ----------------
                else:
                    if open_positions >= max_open:
                        continue
                    if only_buy_if_not_in_pos and in_pos:
                        continue

                    # Keep a reserve
                    if v_free_quote < (stake + eur_reserve):
                        continue

                    ok, reason = should_buy(sig, in_pos, cfg)
                    if not ok:
                        continue

                    try:
                        _, filled, avg, cost, buy_fee, _ticker = place_market_buy(
                            ex, market, stake, taker_fee_pct
                        )
                    except Exception as e:
                        LOG.error(f"{market}: buy failed: {e}")
                        continue

                    # register bot position
                    state.setdefault("positions", {})
                    state["positions"][market] = {
                        "opened_by_bot": True,
                        "base_amount": filled,
                        "entry_price": avg,
                        "entry_ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                        "entry_cost_quote": cost,
                        "entry_fee_quote": buy_fee,
                    }

                    state.setdefault("cooldown", {})
                    state["cooldown"][market] = time.time() + cooldown_min * 60
                    save_state(state)

                    open_positions += 1
                    v_free_quote -= float(cost)  # spend quote (approx)

                    row = {
                        "ts": now_iso(),
                        "market": market,
                        "mode": "trend",
                        "side": "BUY",
                        "price": avg,
                        "base_amount": filled,
                        "fees_quote": buy_fee,
                        "spread_pct": spr,
                        "net_pnl_quote": 0.0,
                        "holding_time_min": 0.0,
                        "reason": reason,
                    }
                    append_tx(row)

                    LOG.info(
                        f"{market} BOUGHT for {quote} {stake:,.2f} @ {avg:,.6f}, spread {spr:.3f}%"
                        .replace(",", "X").replace(".", ",").replace("X", ".")
                    )

            time.sleep(sleep_s)

        except KeyboardInterrupt:
            LOG.info("Stopped by user.")
            break
        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
