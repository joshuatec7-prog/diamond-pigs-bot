#!/usr/bin/env python3
# Diamond-Pigs-stijl bot (Trend v2 compat) - Bitvavo via ccxt
# - Config in NL keys (citaat/symbolen/tijdsbestek/strategie/risico/fees/logging)
# - Accepteert WAAR/waar/true/True/ja/1/aan voor booleans
# - Accepteert komma-decimalen (0,12) én punt-decimalen (0.12)
# - Koopt/verkoopt alleen posities die de bot zelf geopend heeft (staat.json)
# - Netto PnL inclusief fees gelogd per SELL + totaal in staat.json
# - Veilige balance handling (retries/backoff)
#
# Vereisten: ccxt, pandas, pyyaml, python-dotenv
# Env:
#   BITVAVO_API_KEY, BITVAVO_API_SECRET
#   (aanrader) BITVAVO_OPERATOR_ID  -> nodig voor ccxt createOrder bij Bitvavo (als jouw ccxt dit eist)
# Optioneel:
#   CFG_FILE=/opt/render/project/src/config.yaml
#   STATE_FILE=/opt/render/project/src/staat.json

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
        return s in ("true", "waar", "yes", "ja", "1", "aan", "on", "y")
    return default

def to_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("%", "")
        # accepteer "0,12" als 0.12
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
        s = v.strip()
        try:
            return int(float(s.replace(",", ".")))
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
# fallback: sommige users hebben conFig.yaml of metFig.yaml per ongeluk
FALLBACK_CFGS = [ROOT / "conFig.yaml", ROOT / "metFig.yaml", ROOT / "config.yaml"]

STATE_FILE = Path(os.getenv("STATE_FILE", str(ROOT / "staat.json")))
TX_CSV = ROOT / "transacties.csv"

LOG = logging.getLogger("diamond")


def setup_logging(level: str):
    level = (level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------ Config ------------------

def load_cfg() -> dict:
    cfg_path = CFG_FILE
    if not cfg_path.exists():
        for p in FALLBACK_CFGS:
            if p.exists():
                cfg_path = p
                break

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config niet gevonden. Verwacht {CFG_FILE} of fallback in {FALLBACK_CFGS}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Normalize: accepteer ook Engelse keys als iemand die gebruikt
    # (maar bot werkt primair NL)
    cfg = dict(raw)

    # alias keys
    if "quote" in cfg and "citaat" not in cfg:
        cfg["citaat"] = cfg["quote"]
    if "symbols" in cfg and "symbolen" not in cfg:
        cfg["symbolen"] = cfg["symbols"]
    if "timeframe" in cfg and "tijdsbestek" not in cfg:
        cfg["tijdsbestek"] = cfg["timeframe"]
    if "strategy" in cfg and "strategie" not in cfg:
        cfg["strategie"] = cfg["strategy"]
    if "risk" in cfg and "risico" not in cfg:
        cfg["risico"] = cfg["risk"]
    if "fees" in cfg and "vergoedingen" not in cfg:
        cfg["vergoedingen"] = cfg["fees"]
    if "logging" in cfg and "logboek" not in cfg:
        cfg["logboek"] = cfg["logging"]

    # defaults
    cfg.setdefault("citaat", "EUR")
    cfg.setdefault("symbolen", ["BTC", "ETH"])
    cfg.setdefault("tijdsbestek", "15m")
    cfg.setdefault("strategie", {})
    cfg.setdefault("risico", {})
    cfg.setdefault("vergoedingen", {})
    cfg.setdefault("logboek", {})

    # strategie defaults
    st = cfg["strategie"]
    st.setdefault("sma_fast", 20)
    st.setdefault("sma_slow", 50)
    st.setdefault("rsi_len", 14)
    st.setdefault("rsi_buy_min", 55)
    st.setdefault("atr_len", 14)
    st.setdefault("atr_tp_mult", 2.2)
    st.setdefault("atr_sl_mult", 1.4)
    st.setdefault("exit_on_trend_break", True)

    # risico defaults
    rk = cfg["risico"]
    rk.setdefault("vaste_inzetkoers", 15)  # EUR per trade, winst niet herbelegd
    rk.setdefault("max_open_posities", 6)
    rk.setdefault("alleen_kopen_als_je_niet_in_de_positie", True)
    rk.setdefault("afkoeltijd_minuten", 30)
    rk.setdefault("max_spread_pct", 0.12)

    # vergoedingen defaults (taker fee)
    fees = cfg["vergoedingen"]
    fees.setdefault("taker_fee_pct", 0.25)

    # logging defaults
    lg = cfg["logboek"]
    lg.setdefault("niveau", "INFO")
    lg.setdefault("loop_sleep_seconds", 30)
    lg.setdefault("candles_limit", 300)

    # parse & coerce types
    st["sma_fast"] = to_int(st.get("sma_fast"), 20)
    st["sma_slow"] = to_int(st.get("sma_slow"), 50)
    st["rsi_len"] = to_int(st.get("rsi_len"), 14)
    st["rsi_buy_min"] = to_float(st.get("rsi_buy_min"), 55)
    st["atr_len"] = to_int(st.get("atr_len"), 14)
    st["atr_tp_mult"] = to_float(st.get("atr_tp_mult"), 2.2)
    st["atr_sl_mult"] = to_float(st.get("atr_sl_mult"), 1.4)
    st["exit_on_trend_break"] = to_bool(st.get("exit_on_trend_break"), True)

    rk["vaste_inzetkoers"] = to_float(rk.get("vaste_inzetkoers"), 15)
    rk["max_open_posities"] = to_int(rk.get("max_open_posities"), 6)
    rk["alleen_kopen_als_je_niet_in_de_positie"] = to_bool(rk.get("alleen_kopen_als_je_niet_in_de_positie"), True)
    rk["afkoeltijd_minuten"] = to_int(rk.get("afkoeltijd_minuten"), 30)
    rk["max_spread_pct"] = to_float(rk.get("max_spread_pct"), 0.12)

    fees["taker_fee_pct"] = to_float(fees.get("taker_fee_pct"), 0.25)

    lg["niveau"] = str(lg.get("niveau", "INFO")).upper()
    lg["loop_sleep_seconds"] = to_int(lg.get("loop_sleep_seconds"), 30)
    lg["candles_limit"] = to_int(lg.get("candles_limit"), 300)

    cfg["_cfg_path"] = str(cfg_path)
    return cfg


# ------------------ State & CSV ------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "positions": {},   # market -> {base_amount, entry_price, entry_ts, entry_cost_eur, entry_fee_eur}
        "cooldown": {},    # market -> next_allowed_ts (unix)
        "pnl_eur": 0.0,    # cumulative net realized pnl
        "trades": 0,       # number of closed trades (SELL)
        "wins": 0
    }

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def ensure_csv_header():
    if TX_CSV.exists():
        return
    cols = [
        "ts",
        "markt",
        "modus",
        "zijde",
        "prijs",
        "bedrag_basis",
        "kosten_eur",
        "spread_pct",
        "netto_pnl_eur",
        "holding_time_min",
        "reden",
    ]
    pd.DataFrame([], columns=cols).to_csv(TX_CSV, index=False)

def append_tx(row: dict):
    ensure_csv_header()
    df = pd.DataFrame([row])
    df.to_csv(TX_CSV, index=False, mode="a", header=False)


# ------------------ Exchange ------------------

def make_exchange():
    api_key = os.getenv("BITVAVO_API_KEY", "").strip()
    api_secret = os.getenv("BITVAVO_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("BITVAVO_API_KEY/SECRET ontbreken in environment variables.")

    ex = ccxt.bitvavo({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    # operatorId fix (sommige ccxt versies eisen dit)
    opid = os.getenv("BITVAVO_OPERATOR_ID", "").strip()
    if opid:
        try:
            ex.options = ex.options or {}
            ex.options["operatorId"] = int(opid)
        except Exception:
            LOG.warning("BITVAVO_OPERATOR_ID kon niet als int gezet worden. Laat leeg of zet een nummer.")

    return ex


def safe_fetch_balance(ex, retries=5, base_sleep=1.0):
    last_err = None
    for i in range(retries):
        try:
            return ex.fetch_balance()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** i)
            LOG.warning(f"Balance ophalen mislukt ({i+1}/{retries}): {e}. Wacht {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Balance ophalen faalt na {retries} pogingen: {last_err}")


# ------------------ Trading logic ------------------

def market_symbol(base: str, quote: str) -> str:
    return f"{base}/{quote}"

def get_free_quote(balance: dict, quote: str) -> float:
    # ccxt balance: balance["free"][quote]
    try:
        free = balance.get("free", {}).get(quote, 0.0)
        return float(free or 0.0)
    except Exception:
        return 0.0

def fetch_candles(ex, market: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) < 50:
        raise RuntimeError("Te weinig candle data.")
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

def round_down(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def market_limits(ex, market: str):
    mk = ex.market(market)
    amount_step = 0.0
    min_amount = 0.0
    min_cost = 0.0

    try:
        amount_step = float(mk.get("precision", {}).get("amount", 0) or 0)
        # ccxt precision amount is decimals, not step; safer: use limits
    except Exception:
        amount_step = 0.0

    try:
        min_amount = float(mk.get("limits", {}).get("amount", {}).get("min", 0) or 0)
        min_cost = float(mk.get("limits", {}).get("cost", {}).get("min", 0) or 0)
    except Exception:
        min_amount = 0.0
        min_cost = 0.0

    return min_amount, min_cost

def compute_signal(df: pd.DataFrame, cfg: dict) -> dict:
    st = cfg["strategie"]
    sma_fast = st["sma_fast"]
    sma_slow = st["sma_slow"]
    rsi_len = st["rsi_len"]
    rsi_buy_min = st["rsi_buy_min"]
    atr_len = st["atr_len"]

    closes = df["close"]
    df["sma_fast"] = sma(closes, sma_fast)
    df["sma_slow"] = sma(closes, sma_slow)
    df["rsi"] = rsi(closes, rsi_len)
    df["atr"] = atr(df, atr_len)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_up = (last["sma_fast"] > last["sma_slow"])
    cross_up = (prev["sma_fast"] <= prev["sma_slow"]) and (last["sma_fast"] > last["sma_slow"])
    rsi_ok = (last["rsi"] >= rsi_buy_min)

    return {
        "trend_up": bool(trend_up),
        "cross_up": bool(cross_up),
        "rsi": float(last["rsi"]),
        "rsi_ok": bool(rsi_ok),
        "atr": float(last["atr"]) if not math.isnan(float(last["atr"])) else 0.0,
        "last_close": float(last["close"]),
        "trend_break": bool(last["sma_fast"] < last["sma_slow"]),
    }

def should_buy(signal: dict, in_position: bool) -> (bool, str):
    if in_position:
        return False, "In positie"
    if not signal["trend_up"]:
        return False, "Geen trend omhoog"
    if not signal["rsi_ok"]:
        return False, f"RSI {signal['rsi']:.1f} < drempel"
    # cross_up niet verplicht (anders te weinig trades), maar helpt kwaliteit.
    return True, "Signaal"

def should_sell(signal: dict, pos: dict, cfg: dict, last_price: float) -> (bool, str, float, float):
    st = cfg["strategie"]
    atr_tp_mult = st["atr_tp_mult"]
    atr_sl_mult = st["atr_sl_mult"]
    exit_on_trend_break = st["exit_on_trend_break"]

    entry = float(pos["entry_price"])
    atr_val = float(signal.get("atr", 0.0) or 0.0)
    if atr_val <= 0:
        # fallback: geen ATR -> geen ATR exits
        atr_val = max(entry * 0.002, 0.01)

    tp = entry + atr_tp_mult * atr_val
    sl = entry - atr_sl_mult * atr_val

    if last_price >= tp:
        return True, "ATR_TP", tp, sl
    if last_price <= sl:
        return True, "ATR_SL", tp, sl
    if exit_on_trend_break and signal.get("trend_break", False):
        return True, "TREND_BREAK", tp, sl
    return False, "Hold", tp, sl

def fee_estimate_eur(quote_amount: float, taker_fee_pct: float) -> float:
    return quote_amount * (taker_fee_pct / 100.0)

def place_market_buy(ex, market: str, quote_amount: float, taker_fee_pct: float):
    # Koop met quote-amount: bereken base amount via ticker ask (benadering)
    ticker = fetch_ticker(ex, market)
    ask = float(ticker.get("ask") or 0.0)
    if ask <= 0:
        raise RuntimeError("Geen ask prijs beschikbaar.")

    # Netto invest = quote_amount (bot gebruikt vaste inzet). Fees zitten erbovenop in pnl berekening.
    base_amount = quote_amount / ask

    # respecteer minima
    min_amount, min_cost = market_limits(ex, market)
    if min_cost and quote_amount < min_cost:
        raise RuntimeError(f"Quote inzet {quote_amount:.2f} lager dan min cost {min_cost:.2f}")

    if min_amount and base_amount < min_amount:
        raise RuntimeError(f"Base amount {base_amount:.8f} lager dan min amount {min_amount}")

    # Plaats order
    order = ex.create_order(market, "market", "buy", base_amount)

    # Real fill info: probeer avg/filled uit order, anders benadering
    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or ask)
    cost = float(order.get("cost") or (filled * avg))

    fee_eur = fee_estimate_eur(cost, taker_fee_pct)

    return order, filled, avg, cost, fee_eur, ticker

def place_market_sell(ex, market: str, base_amount: float, taker_fee_pct: float):
    ticker = fetch_ticker(ex, market)
    bid = float(ticker.get("bid") or 0.0)
    if bid <= 0:
        raise RuntimeError("Geen bid prijs beschikbaar.")

    order = ex.create_order(market, "market", "sell", base_amount)

    filled = float(order.get("filled") or base_amount)
    avg = float(order.get("average") or bid)
    proceeds = float(order.get("cost") or (filled * avg))  # ccxt 'cost' is quote ontvangen bij sell

    fee_eur = fee_estimate_eur(proceeds, taker_fee_pct)

    return order, filled, avg, proceeds, fee_eur, ticker


# ------------------ Main loop ------------------

def main():
    load_dotenv()

    cfg = load_cfg()
    setup_logging(cfg["logboek"]["niveau"])

    LOG.info(f"Trend v2 bot gestart. Config={cfg.get('_cfg_path')}")
    LOG.info(
        "Config booleans | "
        f"exit_on_trend_break={cfg['strategie']['exit_on_trend_break']}, "
        f"alleen_kopen={cfg['risico']['alleen_kopen_als_je_niet_in_de_positie']}"
    )

    ex = make_exchange()
    state = load_state()
    ensure_csv_header()

    quote = str(cfg["citaat"]).upper()
    bases = list(cfg["symbolen"])
    timeframe = str(cfg["tijdsbestek"])
    candles_limit = int(cfg["logboek"]["candles_limit"])

    fixed_stake = float(cfg["risico"]["vaste_inzetkoers"])
    max_open = int(cfg["risico"]["max_open_posities"])
    only_buy_if_not_in_pos = bool(cfg["risico"]["alleen_kopen_als_je_niet_in_de_positie"])
    cooldown_min = int(cfg["risico"]["afkoeltijd_minuten"])
    max_spread_pct = float(cfg["risico"]["max_spread_pct"])

    taker_fee_pct = float(cfg["vergoedingen"]["taker_fee_pct"])

    sleep_s = int(cfg["logboek"]["loop_sleep_seconds"])

    while True:
        try:
            balance = safe_fetch_balance(ex, retries=5, base_sleep=1.0)
            free_quote = get_free_quote(balance, quote)
            LOG.info(f"Vrij {quote}: {free_quote:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            open_positions = len(state.get("positions", {}))
            # Loop over markten
            for base in bases:
                market = market_symbol(base, quote)

                # cooldown
                now = time.time()
                next_ok = float(state.get("cooldown", {}).get(market, 0) or 0)
                if now < next_ok:
                    continue

                in_pos = market in state.get("positions", {})

                # Spread check (altijd)
                try:
                    t = fetch_ticker(ex, market)
                    spr = calc_spread_pct(t)
                except Exception as e:
                    LOG.warning(f"{market}: ticker ophalen mislukt: {e}")
                    continue

                if spr > max_spread_pct:
                    # te brede spread
                    continue

                # Candles + signal
                try:
                    df = fetch_candles(ex, market, timeframe, candles_limit)
                    sig = compute_signal(df, cfg)
                except Exception as e:
                    LOG.warning(f"{market}: candlestick ophalen mislukt: {e}")
                    continue

                last_price = float(sig["last_close"])

                # SELL only if bot position exists
                if in_pos:
                    pos = state["positions"][market]
                    do_sell, reason, tp, sl = should_sell(sig, pos, cfg, last_price)
                    if do_sell:
                        base_amount = float(pos["base_amount"])
                        try:
                            _, filled, avg, proceeds, sell_fee, ticker = place_market_sell(ex, market, base_amount, taker_fee_pct)
                        except Exception as e:
                            LOG.error(f"{market}: verkopen mislukt: {e}")
                            continue

                        entry_cost = float(pos.get("entry_cost_eur", 0.0))
                        entry_fee = float(pos.get("entry_fee_eur", 0.0))

                        # Netto pnl: proceeds - entry_cost - fees
                        net_pnl = (proceeds - entry_cost) - (entry_fee + sell_fee)

                        # holding time
                        entry_ts = pos.get("entry_ts")
                        hold_min = 0.0
                        if entry_ts:
                            try:
                                t0 = datetime.fromisoformat(entry_ts)
                                dt = datetime.now(timezone.utc).astimezone() - t0
                                hold_min = dt.total_seconds() / 60.0
                            except Exception:
                                hold_min = 0.0

                        # update state
                        state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + float(net_pnl)
                        state["trades"] = int(state.get("trades", 0)) + 1
                        if net_pnl > 0:
                            state["wins"] = int(state.get("wins", 0)) + 1

                        # cooldown
                        state["cooldown"][market] = time.time() + cooldown_min * 60

                        # remove position
                        del state["positions"][market]
                        save_state(state)

                        # log + csv
                        row = {
                            "ts": now_iso(),
                            "markt": market,
                            "modus": "trend",
                            "zijde": "VERKOPEN",
                            "prijs": avg,
                            "bedrag_basis": filled,
                            "kosten_eur": (entry_fee + sell_fee),
                            "spread_pct": spr,
                            "netto_pnl_eur": net_pnl,
                            "holding_time_min": hold_min,
                            "reden": reason,
                        }
                        append_tx(row)

                        winrate = (state["wins"] / state["trades"] * 100.0) if state["trades"] else 0.0
                        LOG.info(
                            f"{market} verkocht. Netto PnL €{net_pnl:,.2f}, totaal €{state['pnl_eur']:,.2f}, "
                            f"transacties {state['trades']}, winrate {winrate:.1f}%, hold {hold_min:.1f} min, reden {reason}"
                            .replace(",", "X").replace(".", ",").replace("X", ".")
                        )

                # BUY
                else:
                    if open_positions >= max_open:
                        continue
                    if only_buy_if_not_in_pos and in_pos:
                        continue
                    if free_quote < fixed_stake:
                        continue

                    ok, reason = should_buy(sig, in_pos)
                    if not ok:
                        continue

                    # koop
                    try:
                        _, filled, avg, cost, buy_fee, ticker = place_market_buy(ex, market, fixed_stake, taker_fee_pct)
                    except Exception as e:
                        LOG.error(f"{market}: kopen mislukt: {e}")
                        continue

                    # register bot position
                    state["positions"][market] = {
                        "base_amount": filled,
                        "entry_price": avg,
                        "entry_ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                        "entry_cost_eur": cost,
                        "entry_fee_eur": buy_fee,
                    }
                    state["cooldown"][market] = time.time() + cooldown_min * 60
                    save_state(state)

                    row = {
                        "ts": now_iso(),
                        "markt": market,
                        "modus": "trend",
                        "zijde": "KOPEN",
                        "prijs": avg,
                        "bedrag_basis": filled,
                        "kosten_eur": buy_fee,
                        "spread_pct": spr,
                        "netto_pnl_eur": 0.0,
                        "holding_time_min": 0.0,
                        "reden": reason,
                    }
                    append_tx(row)

                    LOG.info(
                        f"{market} gekocht voor netto €{fixed_stake:,.2f} @ {avg:,.6f}, spread {spr:.3f}%"
                        .replace(",", "X").replace(".", ",").replace("X", ".")
                    )

            time.sleep(sleep_s)

        except KeyboardInterrupt:
            LOG.info("Stop door gebruiker.")
            break
        except Exception as e:
            LOG.error(f"Hoofdlus fout: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
