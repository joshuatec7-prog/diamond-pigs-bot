#!/usr/bin/env python3
import os
import time
import json
import random
import socket
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd
import yaml
import ccxt


# =========================
# Paden / bestanden
# =========================
ROOT = Path(__file__).parent
CFG_FILE = ROOT / os.getenv("CFG_FILE", "metFig.yaml")
STATE_FILE = ROOT / os.getenv("STATE_FILE", "staat.json")
TRADES_CSV = ROOT / os.getenv("TRADES_CSV", "transacties.csv")

# Dashboard
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() in ("1", "true", "yes", "on")
PORT = int(os.getenv("PORT", "10000"))  # Render Web Service gebruikt PORT

TZ = timezone.utc

# Bitvavo ENV
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY", "")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET", "")
BITVAVO_OPERATOR_ID = os.getenv("BITVAVO_OPERATOR_ID", "").strip()

# Fee fallback (taker) als exchange fee niet beschikbaar is
FEE_RATE_FALLBACK = float(os.getenv("FEE_RATE_FALLBACK", "0.0025"))  # 0.25%


# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now(TZ).isoformat()


def pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0 if b else 0.0


def load_cfg() -> dict:
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Config niet gevonden: {CFG_FILE}")
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(s, dict):
                s.setdefault("pnl_eur", 0.0)
                s.setdefault("closed_trades", 0)
                s.setdefault("wins", 0)
                s.setdefault("positions", {})
                s.setdefault("cooldown", {})
                return s
        except Exception:
            pass
    return {
        "pnl_eur": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "positions": {},   # per market: bot-owned positie
        "cooldown": {},    # per market: iso timestamp
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_trade(row: dict) -> None:
    hdr = not TRADES_CSV.exists()
    pd.DataFrame([row]).to_csv(TRADES_CSV, mode="a", index=False, header=hdr)


# =========================
# Indicatoren
# =========================
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
    """
    Average True Range.
    df moet kolommen hebben: high, low, close
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length).mean()


# =========================
# Safe API calling (retries)
# =========================
def _is_temporary_net_error(e: Exception) -> bool:
    msg = str(e).lower()
    needles = [
        "timed out", "timeout",
        "temporarily unavailable",
        "connection reset", "connection aborted", "connection refused",
        "remote disconnected",
        "bad gateway", "service unavailable", "gateway timeout",
        "502", "503", "504",
        "rate limit", "too many requests",
        "network error", "requesttimeout", "datanotavailable",
        "bitvavo get https://api.bitvavo.com",
    ]
    if any(n in msg for n in needles):
        return True
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    return False


def safe_call(fn, *, name: str = "call", retries: int = 8, base_sleep: float = 2.0, max_sleep: float = 60.0):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_temporary_net_error(e):
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.7 + random.random() * 0.6)  # jitter
            logging.warning(f"Tijdelijke API/netwerk fout bij {name} ({attempt}/{retries}): {e}. Wacht {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last


# =========================
# Exchange
# =========================
def make_exchange() -> ccxt.bitvavo:
    if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
        raise RuntimeError("BITVAVO_API_KEY / BITVAVO_API_SECRET ontbreken in Render env vars")

    ex = ccxt.bitvavo({
        "apiKey": BITVAVO_API_KEY,
        "secret": BITVAVO_API_SECRET,
        "enableRateLimit": True,
        "options": {},
    })
    if BITVAVO_OPERATOR_ID:
        ex.options["operatorId"] = BITVAVO_OPERATOR_ID

    ex.timeout = 30000
    return ex


def get_fee_rate(exchange: ccxt.bitvavo) -> float:
    try:
        fees = getattr(exchange, "fees", None) or {}
        tr = (fees.get("trading") or {})
        taker = tr.get("taker", None)
        if taker is not None:
            return float(taker)
    except Exception:
        pass
    return float(FEE_RATE_FALLBACK)


def fetch_balance(exchange: ccxt.bitvavo) -> dict:
    return safe_call(lambda: exchange.fetch_balance(), name="fetch_balance")


def free_eur(balance: dict) -> float:
    eur = balance.get("EUR") or {}
    return float(eur.get("free", 0.0) or 0.0)


def candles_df(exchange: ccxt.bitvavo, market: str, interval: str, limit: int) -> pd.DataFrame:
    raw = safe_call(lambda: exchange.fetch_ohlcv(market, timeframe=interval, limit=limit), name=f"fetch_ohlcv {market}")
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def orderbook_spread_pct(exchange: ccxt.bitvavo, market: str) -> float:
    ob = safe_call(lambda: exchange.fetch_order_book(market, limit=1), name=f"fetch_order_book {market}")
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    bid = float(bids[0][0]) if bids else 0.0
    ask = float(asks[0][0]) if asks else 0.0
    if bid and ask:
        return (ask - bid) / ((ask + bid) / 2) * 100.0
    return 999.0


def last_price(exchange: ccxt.bitvavo, market: str) -> float:
    t = safe_call(lambda: exchange.fetch_ticker(market), name=f"fetch_ticker {market}")
    return float(t["last"])


def place_market_buy(exchange: ccxt.bitvavo, market: str, stake_eur: float, fee_rate: float):
    px = last_price(exchange, market)
    # ccxt market buy: amount in base
    amount_base = max(0.0, (stake_eur / px) * (1.0 - fee_rate))
    order = safe_call(lambda: exchange.create_order(market, "market", "buy", float(amount_base)), name=f"create_order BUY {market}")
    return order, px


def place_market_sell(exchange: ccxt.bitvavo, market: str, amount_base: float):
    order = safe_call(lambda: exchange.create_order(market, "market", "sell", float(amount_base)), name=f"create_order SELL {market}")
    return order


def extract_order_fill(order: dict, fallback_price: float, fee_rate: float):
    filled = float(order.get("filled") or order.get("amount") or 0.0)
    avg = float(order.get("average") or 0.0) or float(order.get("price") or 0.0) or float(fallback_price)

    cost = order.get("cost", None)
    if cost is None:
        cost = filled * avg
    cost = float(cost)

    fee = order.get("fee") or {}
    fee_cost = fee.get("cost", None)
    if fee_cost is None:
        fee_cost = cost * fee_rate
    fee_cost = float(fee_cost)

    return filled, avg, cost, fee_cost


# =========================
# Signal (trend-only) + ATR filter
# =========================
def compute_signals(df: pd.DataFrame, cfg: dict):
    sig = cfg.get("signals", {}) or {}

    use_sma = bool(sig.get("use_sma", True))
    use_rsi = bool(sig.get("use_rsi", True))

    sma_fast = int(sig.get("sma_fast", 20))
    sma_slow = int(sig.get("sma_slow", 50))

    rsi_len = int(sig.get("rsi_len", 14))
    rsi_buy_min = float(sig.get("rsi_buy_min", 58))
    rsi_sell_max = float(sig.get("rsi_sell_max", 48))

    use_atr_filter = bool(sig.get("use_atr_filter", False))
    atr_len = int(sig.get("atr_len", 14))
    min_atr_pct = float(sig.get("min_atr_pct", 0.0))

    # Indicatoren
    if use_sma:
        df["sma_fast"] = sma(df["close"], sma_fast)
        df["sma_slow"] = sma(df["close"], sma_slow)
    if use_rsi:
        df["rsi"] = rsi(df["close"], rsi_len)

    if use_atr_filter:
        df["atr"] = atr(df, atr_len)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100.0
    else:
        df["atr_pct"] = float("nan")

    df = df.dropna().copy()
    if len(df) < 3:
        return df, False, False, 0.0, True  # atr_ok=True bij gebrek aan filter

    last = df.iloc[-1]

    # ATR filter check
    atr_pct_last = float(last.get("atr_pct", float("nan")))
    atr_ok = True
    if use_atr_filter:
        if not (atr_pct_last == atr_pct_last):  # NaN
            atr_ok = False
        elif atr_pct_last < min_atr_pct:
            atr_ok = False

    # Trend-only buy: trend up + RSI momentum + ATR ok
    buy_sig = True
    sell_sig = False

    if use_sma:
        buy_sig &= last["sma_fast"] > last["sma_slow"]
        sell_sig |= last["sma_fast"] < last["sma_slow"]

    if use_rsi:
        buy_sig &= last["rsi"] >= rsi_buy_min
        sell_sig |= last["rsi"] <= rsi_sell_max

    if use_atr_filter:
        buy_sig &= atr_ok

    return df, bool(buy_sig), bool(sell_sig), atr_pct_last, atr_ok


# =========================
# TP/SL/Trailing
# =========================
def tp_sl_check(cur_price: float, pos: dict, prot: dict):
    tp_pct = float(prot.get("take_profit_pct", 5.0))
    sl_pct = float(prot.get("stop_loss_pct", 2.0))
    tr_pct = float(prot.get("trailing_tp_pct", 2.0))

    avg = float(pos["buy_price"])

    tp_hit = pct(cur_price, avg) >= tp_pct
    sl_hit = pct(cur_price, avg) <= -abs(sl_pct)

    if tr_pct > 0 and float(pos.get("peak", 0.0) or 0.0) > 0:
        giveback_hit = pct(cur_price, float(pos["peak"])) <= -abs(tr_pct)
        tp_hit = tp_hit or giveback_hit

    return tp_hit, sl_hit, tp_pct, sl_pct, tr_pct


# =========================
# Risk helpers
# =========================
def open_positions_count(state: dict) -> int:
    return sum(1 for p in state["positions"].values() if p.get("open", False))


def can_buy_market(state: dict, market: str, risk: dict) -> bool:
    pos = state["positions"].get(market, {})
    if bool(risk.get("only_buy_if_not_in_position", True)) and bool(pos.get("open", False)):
        return False

    cd_iso = state["cooldown"].get(market)
    if cd_iso:
        try:
            if datetime.fromisoformat(cd_iso) > datetime.now(TZ):
                return False
        except Exception:
            pass
    return True


# =========================
# Dashboard
# =========================
STATE_CACHE = {"status": {}}


def build_status_snapshot(state: dict) -> dict:
    closed = int(state.get("closed_trades", 0))
    wins = int(state.get("wins", 0))
    winrate = (100.0 * wins / closed) if closed else 0.0

    positions = []
    for market, p in state.get("positions", {}).items():
        if not p.get("open"):
            continue
        positions.append({
            "market": market,
            "amount_base": p.get("amount_base", 0.0),
            "buy_price": p.get("buy_price", 0.0),
            "buy_total_eur": p.get("buy_total_eur", 0.0),
            "opened_ts": p.get("opened_ts", ""),
            "peak": p.get("peak", 0.0),
        })

    return {
        "ts": now_iso(),
        "pnl_eur": round(float(state.get("pnl_eur", 0.0)), 2),
        "closed_trades": closed,
        "wins": wins,
        "winrate_pct": round(winrate, 2),
        "open_positions": positions,
    }


def read_last_trades(n: int = 50):
    if not TRADES_CSV.exists():
        return []
    try:
        df = pd.read_csv(TRADES_CSV)
        if df.empty:
            return []
        return df.tail(n).to_dict(orient="records")
    except Exception:
        return []


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return

        if self.path.startswith("/api/status"):
            data = {
                "status": STATE_CACHE.get("status", {}),
                "trades": read_last_trades(50),
            }
            self._send(200, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                       "application/json; charset=utf-8")
            return

        status = STATE_CACHE.get("status", {})
        trades = read_last_trades(30)

        rows_pos = ""
        for p in status.get("open_positions", []):
            rows_pos += (
                f"<tr><td>{p['market']}</td><td>{p.get('amount_base',0):.8f}</td>"
                f"<td>{p.get('buy_price',0):.8f}</td><td>{p.get('peak',0):.8f}</td>"
                f"<td>{p.get('opened_ts','')}</td></tr>"
            )
        if not rows_pos:
            rows_pos = "<tr><td colspan='5'>Geen</td></tr>"

        rows_tr = ""
        for t in trades:
            rows_tr += (
                f"<tr><td>{t.get('ts','')}</td><td>{t.get('markt','')}</td><td>{t.get('zijde','')}</td>"
                f"<td>{t.get('prijs','')}</td><td>{t.get('netto_pnl_eur','')}</td><td>{t.get('reden','')}</td></tr>"
            )
        if not rows_tr:
            rows_tr = "<tr><td colspan='6'>Nog geen trades</td></tr>"

        html = f"""<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Diamond Bot Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
.small {{ color: #555; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; font-size: 13px; }}
</style>
</head>
<body>
<h1>Diamond Bot Dashboard</h1>
<div class="small">Laatste update: {status.get("ts","")}</div>

<div class="card">
  <div><b>Totaal PnL:</b> € {status.get("pnl_eur", 0.0)}</div>
  <div><b>Gesloten trades:</b> {status.get("closed_trades", 0)}</div>
  <div><b>Winrate:</b> {status.get("winrate_pct", 0.0)}%</div>
</div>

<div class="card">
  <h3>Open posities</h3>
  <table>
    <thead><tr><th>Market</th><th>Amount</th><th>Buy</th><th>Peak</th><th>Opened</th></tr></thead>
    <tbody>{rows_pos}</tbody>
  </table>
</div>

<div class="card">
  <h3>Laatste trades</h3>
  <table>
    <thead><tr><th>ts</th><th>markt</th><th>zijde</th><th>prijs</th><th>netto_pnl_eur</th><th>reden</th></tr></thead>
    <tbody>{rows_tr}</tbody>
  </table>
</div>

<div class="small">API: <a href="/api/status">/api/status</a></div>
</body>
</html>"""
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")


def run_dashboard():
    srv = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    logging.info(f"Dashboard actief op poort {PORT} (/, /api/status, /health)")
    srv.serve_forever()


# =========================
# Main
# =========================
def main():
    cfg = load_cfg()
    state = load_state()

    log_cfg = cfg.get("logging", {}) or {}
    lvl = str(log_cfg.get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exchange = make_exchange()
    fee_rate = get_fee_rate(exchange)

    quote = str(cfg.get("quote", "EUR")).upper()
    symbols = cfg.get("symbols", [])
    timeframe = str(cfg.get("timeframe", "15m"))
    candles_limit = int(log_cfg.get("candles_limit", 400))
    sleep_s = int(log_cfg.get("loop_sleep_seconds", 30))

    risk = cfg.get("risk", {}) or {}
    prot = cfg.get("protection", {}) or {}

    fixed_stake = float(risk.get("fixed_stake_quote", 15))
    stake_fraction = float(risk.get("stake_fraction", 0.0))  # niet gebruikt als fixed_stake > 0

    max_open_positions = int(risk.get("max_open_positions", 2))
    cooldown_minutes = int(risk.get("cooldown_minutes", 60))
    max_spread_pct = float(risk.get("max_spread_pct", 0.12))
    eur_reserve = float(risk.get("eur_reserve", 50.0))

    markets = [f"{s}/{quote}" for s in symbols]

    state.setdefault("positions", {})
    state.setdefault("cooldown", {})
    state.setdefault("pnl_eur", 0.0)
    state.setdefault("closed_trades", 0)
    state.setdefault("wins", 0)

    logging.info("Diamond bot gestart (trend-only + ATR filter + safe retries)")

    if DASHBOARD_ENABLED:
        threading.Thread(target=run_dashboard, daemon=True).start()

    while True:
        try:
            bal = fetch_balance(exchange)
            free = free_eur(bal)
            usable = max(0.0, free - eur_reserve)
            logging.info(f"Vrij EUR: {free:.2f} (bruikbaar: {usable:.2f})")

            STATE_CACHE["status"] = build_status_snapshot(state)

            for market in markets:
                try:
                    # Spread filter
                    spread = orderbook_spread_pct(exchange, market)
                    if spread > max_spread_pct:
                        continue

                    # Candles + signals
                    df = candles_df(exchange, market, timeframe, candles_limit)
                    if df.empty:
                        continue

                    df, buy_sig, sell_sig, atr_pct_last, atr_ok = compute_signals(df, cfg)
                    if df.empty:
                        continue

                    px = float(df.iloc[-1]["close"])

                    pos = state["positions"].get(market, {"open": False})
                    state["positions"].setdefault(market, pos)

                    # Update peak
                    if pos.get("open"):
                        peak = float(pos.get("peak", pos.get("buy_price", px)) or px)
                        if px > peak:
                            pos["peak"] = px

                    # SELL: alleen bot-owned posities
                    if pos.get("open"):
                        tp_hit, sl_hit, tp_pct, sl_pct, tr_pct = tp_sl_check(px, pos, prot)
                        reason = None
                        if sl_hit:
                            reason = "SL"
                        elif tp_hit:
                            reason = "TP"
                        elif sell_sig:
                            reason = "Signaal"

                        if reason:
                            amount_base = float(pos["amount_base"])
                            order = place_market_sell(exchange, market, amount_base)
                            filled, avg, sell_cost, sell_fee = extract_order_fill(order, px, fee_rate)

                            buy_total = float(pos.get("buy_total_eur", 0.0))
                            net_received = sell_cost - sell_fee
                            net_pnl = net_received - buy_total

                            opened_ts = datetime.fromisoformat(pos["opened_ts"])
                            hold_min = (datetime.now(TZ) - opened_ts).total_seconds() / 60.0

                            state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + float(net_pnl)
                            state["closed_trades"] = int(state.get("closed_trades", 0)) + 1
                            if net_pnl > 0:
                                state["wins"] = int(state.get("wins", 0)) + 1
                            winrate = (100.0 * state["wins"] / state["closed_trades"]) if state["closed_trades"] else 0.0

                            append_trade({
                                "ts": now_iso(),
                                "markt": market,
                                "modus": "trend",
                                "zijde": "VERKOPEN",
                                "bedrag_basis": float(filled),
                                "kosten_eur": float(sell_fee),
                                "spread_pct": float(spread),
                                "netto_eur": float(net_received),
                                "prijs": float(avg),
                                "holding_time_min": float(hold_min),
                                "netto_pnl_eur": float(net_pnl),
                                "reden": reason,
                                "atr_pct": float(atr_pct_last) if (atr_pct_last == atr_pct_last) else None,
                            })

                            pos.update({"open": False})
                            state["cooldown"][market] = (datetime.now(TZ) + timedelta(minutes=cooldown_minutes)).isoformat()
                            save_state(state)

                            logging.info(
                                f"{market} verkocht. Netto PnL €{net_pnl:.2f}, totaal €{state['pnl_eur']:.2f}, "
                                f"trades {state['closed_trades']}, winrate {winrate:.1f}%, hold {hold_min:.1f} min"
                            )

                    # BUY: trend-only + ATR
                    else:
                        if not can_buy_market(state, market, risk):
                            continue
                        if open_positions_count(state) >= max_open_positions:
                            continue
                        if not buy_sig:
                            continue
                        if usable <= 0:
                            continue

                        # stake
                        stake = fixed_stake if fixed_stake > 0 else (usable * stake_fraction)
                        stake = float(stake)

                        if stake <= 0 or usable < stake:
                            continue

                        # Koop
                        order, fallback_px = place_market_buy(exchange, market, stake, fee_rate)
                        filled, avg, buy_cost, buy_fee = extract_order_fill(order, fallback_px, fee_rate)
                        buy_total = buy_cost + buy_fee

                        state["positions"][market] = {
                            "open": True,
                            "amount_base": float(filled),
                            "buy_price": float(avg),
                            "buy_cost_eur": float(buy_cost),
                            "buy_fee_eur": float(buy_fee),
                            "buy_total_eur": float(buy_total),
                            "opened_ts": now_iso(),
                            "peak": float(avg),
                        }
                        save_state(state)

                        append_trade({
                            "ts": now_iso(),
                            "markt": market,
                            "modus": "trend",
                            "zijde": "KOPEN",
                            "bedrag_basis": float(filled),
                            "kosten_eur": float(buy_fee),
                            "spread_pct": float(spread),
                            "netto_eur": float(buy_total),
                            "prijs": float(avg),
                            "holding_time_min": 0.0,
                            "netto_pnl_eur": 0.0,
                            "reden": "Signaal",
                            "atr_pct": float(atr_pct_last) if (atr_pct_last == atr_pct_last) else None,
                        })

                        logging.info(
                            f"{market} gekocht. Netto €{buy_total:.2f} @ {avg:.6f}, spread {spread:.3f}%, ATR% {atr_pct_last:.3f} (ok={atr_ok})"
                        )

                except Exception as e:
                    logging.exception(f"Fout bij {market}: {e}")

            STATE_CACHE["status"] = build_status_snapshot(state)
            time.sleep(sleep_s)

        except Exception as e:
            if _is_temporary_net_error(e):
                logging.warning(f"Tijdelijke fout in hoofdloop: {e}. Wacht 20s en ga verder.")
                time.sleep(20)
                continue
            logging.exception(f"Hoofdfout: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
