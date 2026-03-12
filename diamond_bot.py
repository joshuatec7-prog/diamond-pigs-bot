#!/usr/bin/env python3
# Diamond-Pigs AutoScan v3.0 - Bitvavo via ccxt
#
# Functies:
# - Automatische scan van EUR-markten op Bitvavo
# - Selecteert kansrijke coins op volume + spread + trend + RSI + ATR
# - Kan meerdere coins tegelijk openen
# - Verkoopt alleen posities die deze bot zelf heeft geopend
# - E-mail alerts
# - DRY_RUN veilig testen
#
# Requirements: ccxt, pandas, pyyaml, python-dotenv

import os
import time
import json
import math
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd
import yaml
from dotenv import load_dotenv
import ccxt

LOG = logging.getLogger("diamond")


# ------------------ Helpers ------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def to_bool(v, default=False) -> bool:
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


def to_float(v, default=0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return default
    return default


def to_int(v, default=0) -> int:
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


def fmt_eur(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


# ------------------ Files ------------------

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


# ------------------ Alerts ------------------

def send_email_alert(subject: str, message: str):
    enabled = to_bool(os.getenv("EMAIL_ENABLED", "true"), True)
    if not enabled:
        return

    email_to = os.getenv("ALERT_EMAIL_TO", "").strip()
    email_from = os.getenv("ALERT_EMAIL_FROM", "").strip()
    email_password = os.getenv("ALERT_EMAIL_PASSWORD", "").strip()

    if not email_to or not email_from or not email_password:
        LOG.info(f"E-mail niet ingesteld: {subject}")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = email_from
        msg["To"] = email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.starttls()
        server.login(email_from, email_password)
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
    except Exception as e:
        LOG.warning(f"E-mail fout: {e}")


# ------------------ Config ------------------

ALLOWED_TOP_KEYS = {"quote", "timeframe", "scanner", "signals", "risk", "fees", "logging", "alerts"}
ALLOWED_SCANNER_KEYS = {"auto_scan", "top_n_markets", "min_quote_volume", "exclude_bases"}
ALLOWED_SIGNALS_KEYS = {
    "use_sma", "sma_fast", "sma_slow",
    "use_rsi", "rsi_len", "rsi_buy_min", "rsi_buy_max",
    "use_atr", "atr_len", "atr_tp_mult", "atr_sl_mult",
    "exit_on_trend_break",
    "use_atr_filter", "min_atr_pct",
}
ALLOWED_RISK_KEYS = {
    "fixed_stake_quote", "max_open_positions", "only_buy_if_not_in_position",
    "cooldown_minutes", "max_spread_pct", "eur_reserve",
    "dry_run", "skip_log_every_seconds",
}
ALLOWED_FEES_KEYS = {"taker_fee_pct"}
ALLOWED_LOG_KEYS = {"level", "loop_sleep_seconds", "candles_limit"}
ALLOWED_ALERTS_KEYS = {"email_enabled"}


def _assert_no_unknown(section: str, data: Any, allowed: set):
    if not isinstance(data, dict):
        return
    unknown = [k for k in data.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Config error: unknown keys in '{section}': {sorted(unknown)}")


def validate_cfg(cfg: Dict[str, Any]):
    if not cfg["quote"]:
        raise ValueError("Config error: quote may not be empty.")
    if cfg["scanner"]["top_n_markets"] <= 0:
        raise ValueError("Config error: top_n_markets must be > 0.")
    if cfg["scanner"]["min_quote_volume"] < 0:
        raise ValueError("Config error: min_quote_volume must be >= 0.")
    if cfg["signals"]["sma_fast"] <= 0 or cfg["signals"]["sma_slow"] <= 0:
        raise ValueError("Config error: SMA lengths must be > 0.")
    if cfg["signals"]["use_sma"] and cfg["signals"]["sma_fast"] >= cfg["signals"]["sma_slow"]:
        raise ValueError("Config error: sma_fast must be smaller than sma_slow.")
    if cfg["risk"]["fixed_stake_quote"] <= 0:
        raise ValueError("Config error: fixed_stake_quote must be > 0.")
    if cfg["risk"]["max_open_positions"] <= 0:
        raise ValueError("Config error: max_open_positions must be > 0.")
    if cfg["risk"]["max_spread_pct"] <= 0:
        raise ValueError("Config error: max_spread_pct must be > 0.")
    if cfg["logging"]["loop_sleep_seconds"] <= 0:
        raise ValueError("Config error: loop_sleep_seconds must be > 0.")
    if cfg["logging"]["candles_limit"] < 100:
        raise ValueError("Config error: candles_limit must be >= 100.")


def load_cfg() -> Dict[str, Any]:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Config not found: {CFG_FILE}")

    raw = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")

    _assert_no_unknown("root", raw, ALLOWED_TOP_KEYS)

    cfg = dict(raw)
    cfg.setdefault("quote", "EUR")
    cfg.setdefault("timeframe", "15m")
    cfg.setdefault("scanner", {})
    cfg.setdefault("signals", {})
    cfg.setdefault("risk", {})
    cfg.setdefault("fees", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("alerts", {})

    sc = cfg["scanner"]
    sig = cfg["signals"]
    rk = cfg["risk"]
    fees = cfg["fees"]
    lg = cfg["logging"]
    al = cfg["alerts"]

    _assert_no_unknown("scanner", sc, ALLOWED_SCANNER_KEYS)
    _assert_no_unknown("signals", sig, ALLOWED_SIGNALS_KEYS)
    _assert_no_unknown("risk", rk, ALLOWED_RISK_KEYS)
    _assert_no_unknown("fees", fees, ALLOWED_FEES_KEYS)
    _assert_no_unknown("logging", lg, ALLOWED_LOG_KEYS)
    _assert_no_unknown("alerts", al, ALLOWED_ALERTS_KEYS)

    sc.setdefault("auto_scan", True)
    sc.setdefault("top_n_markets", 25)
    sc.setdefault("min_quote_volume", 250000)
    sc.setdefault("exclude_bases", ["EUR", "USDT", "USDC"])

    sig.setdefault("use_sma", True)
    sig.setdefault("sma_fast", 20)
    sig.setdefault("sma_slow", 60)
    sig.setdefault("use_rsi", True)
    sig.setdefault("rsi_len", 14)
    sig.setdefault("rsi_buy_min", 55)
    sig.setdefault("rsi_buy_max", 72)
    sig.setdefault("use_atr", True)
    sig.setdefault("atr_len", 14)
    sig.setdefault("atr_tp_mult", 2.0)
    sig.setdefault("atr_sl_mult", 1.8)
    sig.setdefault("exit_on_trend_break", True)
    sig.setdefault("use_atr_filter", True)
    sig.setdefault("min_atr_pct", 0.35)

    rk.setdefault("fixed_stake_quote", 30.0)
    rk.setdefault("max_open_positions", 5)
    rk.setdefault("only_buy_if_not_in_position", True)
    rk.setdefault("cooldown_minutes", 45)
    rk.setdefault("max_spread_pct", 0.35)
    rk.setdefault("eur_reserve", 50.0)
    rk.setdefault("dry_run", True)
    rk.setdefault("skip_log_every_seconds", 600)

    fees.setdefault("taker_fee_pct", 0.25)

    lg.setdefault("level", "INFO")
    lg.setdefault("loop_sleep_seconds", 60)
    lg.setdefault("candles_limit", 250)

    al.setdefault("email_enabled", True)

    cfg["quote"] = str(cfg["quote"]).upper().strip()
    cfg["timeframe"] = str(cfg["timeframe"]).strip()

    sc["auto_scan"] = to_bool(sc.get("auto_scan"), True)
    sc["top_n_markets"] = to_int(sc.get("top_n_markets"), 25)
    sc["min_quote_volume"] = to_float(sc.get("min_quote_volume"), 250000)
    sc["exclude_bases"] = [str(x).upper().strip() for x in sc.get("exclude_bases", []) if str(x).strip()]

    sig["use_sma"] = to_bool(sig.get("use_sma"), True)
    sig["sma_fast"] = to_int(sig.get("sma_fast"), 20)
    sig["sma_slow"] = to_int(sig.get("sma_slow"), 60)
    sig["use_rsi"] = to_bool(sig.get("use_rsi"), True)
    sig["rsi_len"] = to_int(sig.get("rsi_len"), 14)
    sig["rsi_buy_min"] = to_float(sig.get("rsi_buy_min"), 55)
    sig["rsi_buy_max"] = to_float(sig.get("rsi_buy_max"), 72)
    sig["use_atr"] = to_bool(sig.get("use_atr"), True)
    sig["atr_len"] = to_int(sig.get("atr_len"), 14)
    sig["atr_tp_mult"] = to_float(sig.get("atr_tp_mult"), 2.0)
    sig["atr_sl_mult"] = to_float(sig.get("atr_sl_mult"), 1.8)
    sig["exit_on_trend_break"] = to_bool(sig.get("exit_on_trend_break"), True)
    sig["use_atr_filter"] = to_bool(sig.get("use_atr_filter"), True)
    sig["min_atr_pct"] = to_float(sig.get("min_atr_pct"), 0.35)

    rk["fixed_stake_quote"] = to_float(rk.get("fixed_stake_quote"), 30.0)
    rk["max_open_positions"] = to_int(rk.get("max_open_positions"), 5)
    rk["only_buy_if_not_in_position"] = to_bool(rk.get("only_buy_if_not_in_position"), True)
    rk["cooldown_minutes"] = to_int(rk.get("cooldown_minutes"), 45)
    rk["max_spread_pct"] = to_float(rk.get("max_spread_pct"), 0.35)
    rk["eur_reserve"] = to_float(rk.get("eur_reserve"), 50.0)
    rk["dry_run"] = to_bool(rk.get("dry_run"), True)
    rk["skip_log_every_seconds"] = max(10, to_int(rk.get("skip_log_every_seconds"), 600))

    fees["taker_fee_pct"] = to_float(fees.get("taker_fee_pct"), 0.25)

    lg["level"] = str(lg.get("level", "INFO")).upper()
    lg["loop_sleep_seconds"] = to_int(lg.get("loop_sleep_seconds"), 60)
    lg["candles_limit"] = to_int(lg.get("candles_limit"), 250)

    al["email_enabled"] = to_bool(al.get("email_enabled"), True)

    validate_cfg(cfg)
    cfg["_cfg_path"] = str(CFG_FILE)
    return cfg


# ------------------ State & CSV ------------------

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(st, dict):
                st.setdefault("positions", {})
                st.setdefault("cooldown", {})
                st.setdefault("pnl_quote", 0.0)
                st.setdefault("trades", 0)
                st.setdefault("wins", 0)
                return st
        except Exception:
            pass
    return {"positions": {}, "cooldown": {}, "pnl_quote": 0.0, "trades": 0, "wins": 0}


def save_state(state: Dict[str, Any]):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_csv_header():
    if TX_CSV.exists():
        return
    cols = ["ts", "market", "side", "price", "base_amount", "fees_quote", "spread_pct", "net_pnl_quote", "holding_time_min", "reason", "dry_run"]
    pd.DataFrame([], columns=cols).to_csv(TX_CSV, index=False)


def append_tx(row: Dict[str, Any]):
    ensure_csv_header()
    pd.DataFrame([row]).to_csv(TX_CSV, index=False, mode="a", header=False)


# ------------------ Exchange ------------------

def make_exchange():
    api_key = os.getenv("BITVAVO_API_KEY", "").strip()
    api_secret = os.getenv("BITVAVO_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Missing BITVAVO_API_KEY / BITVAVO_API_SECRET.")

    ex = ccxt.bitvavo({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
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


def get_free_quote(balance: Dict[str, Any], quote: str) -> float:
    try:
        return float((balance.get("free", {}) or {}).get(quote, 0.0) or 0.0)
    except Exception:
        return 0.0


# ------------------ Market helpers ------------------

def fetch_ticker(ex, market: str) -> Dict[str, Any]:
    return ex.fetch_ticker(market)


def calc_spread_pct(ticker: Dict[str, Any]) -> float:
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    if not bid or not ask or bid <= 0:
        return 999.0
    return (ask - bid) / ask * 100.0


def fetch_candles(ex, market: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) < 50:
        raise RuntimeError("Not enough candle data.")
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    if len(df) < 50:
        raise RuntimeError("Not enough clean candle data.")
    return df


def amount_to_precision_safe(ex, market: str, amount: float) -> float:
    try:
        return float(ex.amount_to_precision(market, amount))
    except Exception:
        return float(amount)


def market_limits(ex, market: str) -> Tuple[float, float]:
    mk = ex.market(market)
    min_amount = float((mk.get("limits", {}) or {}).get("amount", {}).get("min", 0) or 0)
    min_cost = float((mk.get("limits", {}) or {}).get("cost", {}).get("min", 0) or 0)
    return min_amount, min_cost


def fee_from_order_or_estimate(order: Dict[str, Any], fallback_quote_amount: float, taker_fee_pct: float) -> float:
    try:
        fee = order.get("fee")
        if isinstance(fee, dict) and fee.get("cost") is not None:
            return float(fee["cost"])
        fees = order.get("fees")
        if isinstance(fees, list) and fees:
            total = 0.0
            for f in fees:
                if isinstance(f, dict) and f.get("cost") is not None:
                    total += float(f["cost"])
            if total > 0:
                return total
    except Exception:
        pass
    return float(fallback_quote_amount) * (float(taker_fee_pct) / 100.0)


def get_scan_markets(ex, cfg: Dict[str, Any]) -> List[str]:
    quote = cfg["quote"]
    sc = cfg["scanner"]
    exclude = set(sc["exclude_bases"])
    min_vol = float(sc["min_quote_volume"])
    top_n = int(sc["top_n_markets"])

    markets = []
    for market, mk in ex.markets.items():
        try:
            if not mk.get("active", True):
                continue
            if mk.get("quote") != quote:
                continue
            base = str(mk.get("base", "")).upper()
            if not base or base in exclude:
                continue
            ticker = fetch_ticker(ex, market)
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            if quote_volume < min_vol:
                continue
            spread_pct = calc_spread_pct(ticker)
            markets.append({
                "market": market,
                "quote_volume": quote_volume,
                "spread_pct": spread_pct,
            })
        except Exception:
            continue

    markets = sorted(markets, key=lambda x: x["quote_volume"], reverse=True)
    return [x["market"] for x in markets[:top_n]]


# ------------------ Signals ------------------

def compute_signal(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    sig = cfg["signals"]
    closes = df["close"]

    if sig["use_sma"]:
        df["sma_fast"] = sma(closes, sig["sma_fast"])
        df["sma_slow"] = sma(closes, sig["sma_slow"])
    else:
        df["sma_fast"] = float("nan")
        df["sma_slow"] = float("nan")

    if sig["use_rsi"]:
        df["rsi"] = rsi(closes, sig["rsi_len"])
    else:
        df["rsi"] = float("nan")

    if sig["use_atr"] or sig["use_atr_filter"]:
        df["atr"] = atr(df, sig["atr_len"])
    else:
        df["atr"] = 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_close = float(last["close"])

    trend_up = True
    trend_break = False
    if sig["use_sma"]:
        trend_up = (last["sma_fast"] > last["sma_slow"])
        trend_break = (last["sma_fast"] < last["sma_slow"])

    rsi_val = float(last["rsi"]) if sig["use_rsi"] and not math.isnan(float(last["rsi"])) else 50.0
    rsi_ok = True
    if sig["use_rsi"]:
        rsi_ok = (rsi_val >= sig["rsi_buy_min"]) and (rsi_val <= sig["rsi_buy_max"])

    atr_val = float(last.get("atr", 0.0)) if not math.isnan(float(last.get("atr", 0.0))) else 0.0
    atr_pct = (atr_val / last_close * 100.0) if last_close > 0 else 0.0
    atr_filter_ok = True
    if sig["use_atr_filter"]:
        atr_filter_ok = atr_pct >= sig["min_atr_pct"]

    momentum_up = last_close > float(prev["close"])

    return {
        "trend_up": bool(trend_up),
        "trend_break": bool(trend_break),
        "rsi": float(rsi_val),
        "rsi_ok": bool(rsi_ok),
        "atr": float(atr_val),
        "atr_pct": float(atr_pct),
        "atr_filter_ok": bool(atr_filter_ok),
        "last_close": float(last_close),
        "momentum_up": bool(momentum_up),
    }


def candidate_score(signal: Dict[str, Any], spread_pct: float) -> float:
    score = 0.0
    if signal["trend_up"]:
        score += 40
    if signal["rsi_ok"]:
        score += 20
    if signal["momentum_up"]:
        score += 10
    if signal["atr_filter_ok"]:
        score += 10
    if spread_pct < 0.20:
        score += 10
    elif spread_pct < 0.35:
        score += 5
    score += min(signal["atr_pct"], 5.0) * 2
    return round(score, 2)


def should_buy(signal: Dict[str, Any], in_position: bool, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    if in_position:
        return False, "Already in position"
    if cfg["signals"]["use_sma"] and not signal["trend_up"]:
        return False, "SKIP no uptrend"
    if cfg["signals"]["use_rsi"] and not signal["rsi_ok"]:
        return False, f"SKIP RSI {signal['rsi']:.1f} outside range"
    if cfg["signals"]["use_atr_filter"] and not signal["atr_filter_ok"]:
        return False, f"SKIP ATR% {signal['atr_pct']:.3f} too low"
    return True, "BUY signal"


def should_sell(signal: Dict[str, Any], pos: Dict[str, Any], cfg: Dict[str, Any], last_price: float) -> Tuple[bool, str, float, float]:
    sig = cfg["signals"]
    entry = float(pos["entry_price"])

    atr_val = float(signal.get("atr", 0.0) or 0.0)
    if atr_val <= 0:
        atr_val = max(entry * 0.002, 0.01)

    tp = entry + sig["atr_tp_mult"] * atr_val
    sl = entry - sig["atr_sl_mult"] * atr_val

    if last_price >= tp:
        return True, "SELL ATR_TP", tp, sl
    if last_price <= sl:
        return True, "SELL ATR_SL", tp, sl
    if sig["exit_on_trend_break"] and signal.get("trend_break", False):
        return True, "SELL TREND_BREAK", tp, sl
    return False, "HOLD", tp, sl


# ------------------ Orders ------------------

def place_market_buy(ex, market: str, stake_quote: float, taker_fee_pct: float, dry_run: bool):
    ticker = fetch_ticker(ex, market)
    ask = float(ticker.get("ask") or 0.0)
    if ask <= 0:
        raise RuntimeError("No ask price.")

    base_amount = stake_quote / ask
    base_amount = amount_to_precision_safe(ex, market, base_amount)

    min_amount, min_cost = market_limits(ex, market)
    if min_cost and stake_quote < min_cost:
        raise RuntimeError(f"Stake {stake_quote:.2f} below min cost {min_cost:.2f}")
    if min_amount and base_amount < min_amount:
        raise RuntimeError(f"Amount {base_amount:.8f} below min amount {min_amount}")

    if dry_run:
        filled = float(base_amount)
        avg = float(ask)
        cost = float(filled * avg)
        fake = {
            "id": "DRYRUN",
            "filled": filled,
            "average": avg,
            "cost": cost,
            "fee": {"cost": cost * (taker_fee_pct / 100.0)},
        }
        fee_q = fee_from_order_or_estimate(fake, cost, taker_fee_pct)
        return fake, filled, avg, cost, fee_q, ticker

    order = ex.create_order(market, "market", "buy", base_amount)
    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or ask)
    cost = float(order.get("cost") or (filled * avg))
    fee_q = fee_from_order_or_estimate(order, cost, taker_fee_pct)
    return order, filled, avg, cost, fee_q, ticker


def place_market_sell(ex, market: str, base_amount: float, taker_fee_pct: float, dry_run: bool):
    ticker = fetch_ticker(ex, market)
    bid = float(ticker.get("bid") or 0.0)
    if bid <= 0:
        raise RuntimeError("No bid price.")

    base_amount = amount_to_precision_safe(ex, market, base_amount)

    if dry_run:
        filled = float(base_amount)
        avg = float(bid)
        proceeds = float(filled * avg)
        fake = {
            "id": "DRYRUN",
            "filled": filled,
            "average": avg,
            "cost": proceeds,
            "fee": {"cost": proceeds * (taker_fee_pct / 100.0)},
        }
        fee_q = fee_from_order_or_estimate(fake, proceeds, taker_fee_pct)
        return fake, filled, avg, proceeds, fee_q, ticker

    order = ex.create_order(market, "market", "sell", base_amount)
    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or bid)
    proceeds = float(order.get("cost") or (filled * avg))
    fee_q = fee_from_order_or_estimate(order, proceeds, taker_fee_pct)
    return order, filled, avg, proceeds, fee_q, ticker


# ------------------ Skip logging rate limiter ------------------

_skip_last: Dict[str, float] = {}


def log_skip(market: str, msg: str, every_s: int):
    key = f"{market}:{msg}"
    now = time.time()
    last = _skip_last.get(key, 0.0)
    if now - last >= every_s:
        _skip_last[key] = now
        LOG.info(f"{market} {msg}")


# ------------------ Main ------------------

def main():
    load_dotenv()

    cfg = load_cfg()
    setup_logging(cfg["logging"]["level"])

    os.environ["EMAIL_ENABLED"] = "true" if cfg["alerts"]["email_enabled"] else "false"

    quote = cfg["quote"]
    timeframe = cfg["timeframe"]

    rk = cfg["risk"]
    fees = cfg["fees"]
    lg = cfg["logging"]

    stake = float(rk["fixed_stake_quote"])
    max_open = int(rk["max_open_positions"])
    only_buy_if_not_in_pos = bool(rk["only_buy_if_not_in_position"])
    cooldown_min = int(rk["cooldown_minutes"])
    max_spread_pct = float(rk["max_spread_pct"])
    eur_reserve = float(rk["eur_reserve"])
    dry_run = bool(rk["dry_run"])
    skip_every = int(rk["skip_log_every_seconds"])

    taker_fee_pct = float(fees["taker_fee_pct"])
    sleep_s = int(lg["loop_sleep_seconds"])
    candles_limit = int(lg["candles_limit"])

    LOG.info(f"Trend bot started. Config={cfg.get('_cfg_path')} DRY_RUN={dry_run}")
    send_email_alert(
        "Bot gestart",
        f"Diamond bot gestart\nConfig: {cfg.get('_cfg_path')}\nDRY_RUN: {dry_run}\nMax open posities: {max_open}"
    )

    ex = make_exchange()
    ex.load_markets()

    state = load_state()
    ensure_csv_header()

    while True:
        try:
            balance = safe_fetch_balance(ex, retries=5, base_sleep=1.0)
            free_quote = get_free_quote(balance, quote)
            LOG.info(f"Free {quote}: {fmt_eur(free_quote)}")

            state.setdefault("positions", {})
            state.setdefault("cooldown", {})

            # Eerst bestaande posities beheren
            existing_markets = list((state.get("positions", {}) or {}).keys())

            for market in existing_markets:
                positions = state.get("positions", {}) or {}
                if market not in positions:
                    continue

                pos = positions[market]
                if not to_bool(pos.get("opened_by_bot", False), False):
                    continue

                try:
                    tkr = fetch_ticker(ex, market)
                    spr = calc_spread_pct(tkr)
                    if spr > max_spread_pct:
                        continue

                    df = fetch_candles(ex, market, timeframe, candles_limit)
                    s = compute_signal(df, cfg)
                    last_price = float(s["last_close"])
                except Exception as e:
                    log_skip(market, f"SELL check failed: {e}", skip_every)
                    continue

                do_sell, reason, tp, sl = should_sell(s, pos, cfg, last_price)
                if not do_sell:
                    continue

                base_amount = float(pos["base_amount"])
                try:
                    order, filled, avg, proceeds, sell_fee, _ = place_market_sell(
                        ex, market, base_amount, taker_fee_pct, dry_run
                    )
                except Exception as e:
                    log_skip(market, f"SELL failed: {e}", skip_every)
                    continue

                entry_cost = float(pos.get("entry_cost_quote", 0.0))
                entry_fee = float(pos.get("entry_fee_quote", 0.0))
                net_pnl = (proceeds - entry_cost) - (entry_fee + sell_fee)

                hold_min = 0.0
                entry_ts = pos.get("entry_ts")
                if entry_ts:
                    try:
                        t0 = datetime.fromisoformat(entry_ts)
                        hold_min = (datetime.now(timezone.utc).astimezone() - t0).total_seconds() / 60.0
                    except Exception:
                        hold_min = 0.0

                state["pnl_quote"] = float(state.get("pnl_quote", 0.0)) + float(net_pnl)
                state["trades"] = int(state.get("trades", 0)) + 1
                if net_pnl > 0:
                    state["wins"] = int(state.get("wins", 0)) + 1

                state["cooldown"][market] = time.time() + cooldown_min * 60
                del state["positions"][market]
                save_state(state)

                row = {
                    "ts": now_iso(),
                    "market": market,
                    "side": "SELL",
                    "price": avg,
                    "base_amount": filled,
                    "fees_quote": (entry_fee + sell_fee),
                    "spread_pct": spr,
                    "net_pnl_quote": net_pnl,
                    "holding_time_min": hold_min,
                    "reason": reason,
                    "dry_run": dry_run,
                }
                append_tx(row)

                winrate = (state["wins"] / state["trades"] * 100.0) if state["trades"] else 0.0
                msg = (
                    f"{market} {reason} | PnL {quote} {fmt_eur(net_pnl)} | totaal {quote} {fmt_eur(state['pnl_quote'])} "
                    f"| trades {state['trades']} | winrate {winrate:.1f}% | hold {hold_min:.1f}m"
                )
                LOG.info(msg)
                send_email_alert(
                    f"SELL {market}",
                    f"Markt: {market}\n"
                    f"Reden: {reason}\n"
                    f"PnL: {quote} {fmt_eur(net_pnl)}\n"
                    f"Totaal PnL: {quote} {fmt_eur(state['pnl_quote'])}\n"
                    f"Trades: {state['trades']}\n"
                    f"Winrate: {winrate:.1f}%\n"
                    f"Holding time: {hold_min:.1f} min\n"
                    f"Dry run: {dry_run}"
                )

                balance = safe_fetch_balance(ex, retries=3, base_sleep=0.8)
                free_quote = get_free_quote(balance, quote)

            # Daarna nieuwe markten scannen
            scan_markets = get_scan_markets(ex, cfg)
            LOG.info(f"Scanner vond {len(scan_markets)} markten")

            candidates = []
            for market in scan_markets:
                positions = state.get("positions", {}) or {}
                in_pos = market in positions
                next_ok = float((state.get("cooldown", {}) or {}).get(market, 0) or 0)

                if only_buy_if_not_in_pos and in_pos:
                    continue

                if time.time() < next_ok:
                    continue

                try:
                    tkr = fetch_ticker(ex, market)
                    spr = calc_spread_pct(tkr)
                    if spr > max_spread_pct:
                        continue

                    df = fetch_candles(ex, market, timeframe, candles_limit)
                    s = compute_signal(df, cfg)
                    ok, reason = should_buy(s, in_pos, cfg)
                    if not ok:
                        continue

                    score = candidate_score(s, spr)

                    candidates.append({
                        "market": market,
                        "score": score,
                        "signal": s,
                        "spread_pct": spr,
                    })
                except Exception as e:
                    log_skip(market, f"SCAN failed: {e}", skip_every)
                    continue

            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            for cand in candidates:
                positions = state.get("positions", {}) or {}
                open_positions = len(positions)

                if open_positions >= max_open:
                    break

                market = cand["market"]
                spr = cand["spread_pct"]

                if free_quote < (stake + eur_reserve):
                    log_skip(
                        market,
                        f"SKIP reserve (free {fmt_eur(free_quote)} < stake+reserve {fmt_eur(stake + eur_reserve)})",
                        skip_every,
                    )
                    break

                try:
                    order, filled, avg, cost, buy_fee, _ = place_market_buy(
                        ex, market, stake, taker_fee_pct, dry_run
                    )
                except Exception as e:
                    log_skip(market, f"BUY failed: {e}", skip_every)
                    continue

                state["positions"][market] = {
                    "opened_by_bot": True,
                    "base_amount": filled,
                    "entry_price": avg,
                    "entry_ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    "entry_cost_quote": cost,
                    "entry_fee_quote": buy_fee,
                    "score_at_entry": cand["score"],
                }
                save_state(state)

                row = {
                    "ts": now_iso(),
                    "market": market,
                    "side": "BUY",
                    "price": avg,
                    "base_amount": filled,
                    "fees_quote": buy_fee,
                    "spread_pct": spr,
                    "net_pnl_quote": 0.0,
                    "holding_time_min": 0.0,
                    "reason": f"BUY score {cand['score']}",
                    "dry_run": dry_run,
                }
                append_tx(row)

                LOG.info(f"{market} BUY | score {cand['score']} | stake {quote} {fmt_eur(stake)} @ {avg:.6f} | spread {spr:.3f}%")
                send_email_alert(
                    f"BUY {market}",
                    f"Markt: {market}\n"
                    f"Score: {cand['score']}\n"
                    f"Stake: {quote} {fmt_eur(stake)}\n"
                    f"Prijs: {avg:.6f}\n"
                    f"Spread: {spr:.3f}%\n"
                    f"Dry run: {dry_run}"
                )

                balance = safe_fetch_balance(ex, retries=3, base_sleep=0.8)
                free_quote = get_free_quote(balance, quote)

            time.sleep(sleep_s)

        except KeyboardInterrupt:
            LOG.info("Stopped by user.")
            break
        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            send_email_alert("Bot fout", str(e))
            time.sleep(10)


if __name__ == "__main__":
    main()
