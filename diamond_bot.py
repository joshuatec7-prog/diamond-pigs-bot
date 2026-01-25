# diamond_bot.py
import os, time, json, math, logging, traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

import ccxt

# =========================
# Files / paths
# =========================
ROOT = Path(__file__).parent
DEFAULT_CFG = ROOT / "config.yaml"
FALLBACK_CFG = ROOT / "metFig.yaml"  # legacy name some repos use
STATE_FILE = ROOT / "staat.json"     # keep your Dutch filename
TRADES_CSV = ROOT / "transacties.csv"

# =========================
# Indicators
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

# =========================
# Helpers
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def load_cfg() -> dict:
    cfg_path = Path(os.getenv("CFG_FILE", str(DEFAULT_CFG)))
    if not cfg_path.exists() and FALLBACK_CFG.exists():
        cfg_path = FALLBACK_CFG

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_cfg_path"] = str(cfg_path)
    return cfg

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "positions": {},         # only positions opened by THIS bot
        "cooldown": {},          # per symbol next allowed buy time
        "pnl_eur": 0.0,
        "stats": {
            "closed": 0,
            "wins": 0,
            "losses": 0,
        },
        "last_balance": {"EUR_free": None, "ts": None},
    }

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def ensure_trades_header():
    if TRADES_CSV.exists():
        return
    cols = [
        "ts",
        "market",
        "side",
        "reason",
        "price",
        "amount_base",
        "quote_gross_eur",
        "fee_eur",
        "spread_pct",
        "atr",
        "tp_price",
        "sl_price",
        "holding_time_min",
        "netto_pnl_eur",
        "totaal_pnl_eur",
        "winrate_pct",
    ]
    pd.DataFrame([], columns=cols).to_csv(TRADES_CSV, index=False)

def append_trade(row: dict):
    ensure_trades_header()
    df = pd.DataFrame([row])
    df.to_csv(TRADES_CSV, index=False, mode="a", header=False)

def pct(a: float, b: float) -> float:
    return 0.0 if b == 0 else (a / b) * 100.0

# =========================
# Exchange
# =========================
def make_exchange(cfg: dict) -> ccxt.Exchange:
    load_dotenv()

    api_key = os.getenv("BITVAVO_API_KEY", "")
    api_secret = os.getenv("BITVAVO_API_SECRET", "")
    operator_id = os.getenv("BITVAVO_OPERATOR_ID", "").strip()

    ex_params = {
        "enableRateLimit": True,
    }
    if api_key and api_secret:
        ex_params["apiKey"] = api_key
        ex_params["secret"] = api_secret

    # Bitvavo in ccxt now requires operatorId for createOrder
    if operator_id:
        try:
            ex_params["options"] = {"operatorId": int(operator_id)}
        except Exception:
            ex_params["options"] = {"operatorId": operator_id}

    ex = ccxt.bitvavo(ex_params)
    return ex

def safe_fetch_balance(ex: ccxt.Exchange, state: dict, logger: logging.Logger) -> dict | None:
    try:
        bal = ex.fetch_balance()
        eur = bal.get("EUR") or {}
        eur_free = safe_float(eur.get("free"), None)
        if eur_free is not None:
            state["last_balance"] = {"EUR_free": eur_free, "ts": now_utc().isoformat()}
            return {"EUR_free": eur_free}
        return None
    except Exception as e:
        # Safe fallback: use cached last_balance
        cached = state.get("last_balance") or {}
        eur_free = cached.get("EUR_free")
        if eur_free is not None:
            logger.warning(f"Balance fetch failed, using cached EUR_free={eur_free}. Error: {e}")
            return {"EUR_free": float(eur_free)}
        logger.error(f"Balance fetch failed and no cache available. Error: {e}")
        return None

# =========================
# Strategy logic (Trend v2)
# =========================
def should_buy_trend_v2(df: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    s = cfg["strategy"]
    # Trend filter
    sma_fast = sma(df["close"], int(s["sma_fast"]))
    sma_slow = sma(df["close"], int(s["sma_slow"]))
    trend_up = sma_fast.iloc[-1] > sma_slow.iloc[-1]

    # Pullback entry: previous close below sma_fast, current close above sma_fast
    prev_close = df["close"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    prev_sma_fast = sma_fast.iloc[-2]
    curr_sma_fast = sma_fast.iloc[-1]
    cross_up = (prev_close < prev_sma_fast) and (curr_close > curr_sma_fast)

    # RSI filter
    r = rsi(df["close"], int(s["rsi_len"])).iloc[-1]
    rsi_ok = r >= float(s["rsi_buy_min"])

    # Volatility / noise filter: require ATR available
    a = atr(df, int(s["atr_len"])).iloc[-1]
    atr_ok = not (math.isnan(a) or a <= 0)

    if trend_up and cross_up and rsi_ok and atr_ok:
        return True, "TrendCrossUp"
    return False, "NoSignal"

def should_sell(df: pd.DataFrame, pos: dict, cfg: dict) -> tuple[bool, str, float, float, float]:
    s = cfg["strategy"]
    a = atr(df, int(s["atr_len"])).iloc[-1]
    if math.isnan(a) or a <= 0:
        a = safe_float(pos.get("atr"), 0.0)

    entry = float(pos["entry_price"])
    tp = entry + float(s["atr_tp_mult"]) * a
    sl = entry - float(s["atr_sl_mult"]) * a

    last = float(df["close"].iloc[-1])

    if last >= tp:
        return True, "TakeProfit_ATR", a, tp, sl
    if last <= sl:
        return True, "StopLoss_ATR", a, tp, sl

    # Optional: exit if trend breaks down
    sma_fast = sma(df["close"], int(s["sma_fast"])).iloc[-1]
    sma_slow = sma(df["close"], int(s["sma_slow"])).iloc[-1]
    if bool(s.get("exit_on_trend_break", True)) and sma_fast < sma_slow:
        return True, "TrendBreak", a, tp, sl

    return False, "Hold", a, tp, sl

# =========================
# Execution / risk
# =========================
def spread_pct_from_ticker(t: dict) -> float:
    bid = safe_float(t.get("bid"), 0.0)
    ask = safe_float(t.get("ask"), 0.0)
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0

def calc_fee_eur(cfg: dict, quote_amount_eur: float) -> float:
    fee_pct = float(cfg["fees"]["taker_fee_pct"])
    return quote_amount_eur * (fee_pct / 100.0)

def can_open_new_position(symbol: str, state: dict, cfg: dict, eur_free: float) -> tuple[bool, str]:
    r = cfg["risk"]
    positions = state.get("positions", {})
    max_open = int(r["max_open_positions"])
    if len(positions) >= max_open:
        return False, "MaxOpenPositions"

    # cooldown
    cd = state.get("cooldown", {}).get(symbol)
    if cd:
        try:
            if now_utc().timestamp() < float(cd):
                return False, "Cooldown"
        except Exception:
            pass

    # avoid multiple simultaneous buys per coin
    if bool(r.get("only_buy_if_not_in_position", True)) and symbol in positions:
        return False, "AlreadyInPosition"

    stake = float(r["fixed_stake_quote"])
    if stake <= 0:
        return False, "StakeNotSet"
    if eur_free < stake:
        return False, "InsufficientEUR"

    return True, "OK"

def place_market_buy(ex: ccxt.Exchange, market: str, stake_eur: float) -> dict:
    # Use ticker ask to compute base amount
    t = ex.fetch_ticker(market)
    ask = safe_float(t.get("ask"), 0.0) or safe_float(t.get("last"), 0.0)
    if ask <= 0:
        raise RuntimeError("No valid ask/last price for buy")

    amount_base = stake_eur / ask
    # precision
    amount_base = float(ex.amount_to_precision(market, amount_base))
    if amount_base <= 0:
        raise RuntimeError("Computed amount_base <= 0")

    order = ex.create_order(market, "market", "buy", amount_base)
    return {"order": order, "ticker": t, "price": ask, "amount_base": amount_base}

def place_market_sell(ex: ccxt.Exchange, market: str, amount_base: float) -> dict:
    amount_base = float(ex.amount_to_precision(market, amount_base))
    if amount_base <= 0:
        raise RuntimeError("Sell amount_base <= 0")

    t = ex.fetch_ticker(market)
    bid = safe_float(t.get("bid"), 0.0) or safe_float(t.get("last"), 0.0)
    if bid <= 0:
        raise RuntimeError("No valid bid/last price for sell")

    order = ex.create_order(market, "market", "sell", amount_base)
    return {"order": order, "ticker": t, "price": bid, "amount_base": amount_base}

# =========================
# Main loop
# =========================
def setup_logging(cfg: dict) -> logging.Logger:
    lvl = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("diamond_bot")

def fetch_ohlcv_df(ex: ccxt.Exchange, market: str, timeframe: str, limit: int) -> pd.DataFrame:
    o = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def main():
    cfg = load_cfg()
    logger = setup_logging(cfg)

    logger.info(f"Trend v2 bot started. Config={cfg.get('_cfg_path')}")
    ensure_trades_header()

    ex = make_exchange(cfg)

    # If operatorId missing, allow read-only mode (no trading)
    operator_ok = bool((ex.options or {}).get("operatorId"))
    keys_ok = bool(getattr(ex, "apiKey", "") and getattr(ex, "secret", ""))

    if not keys_ok:
        logger.warning("No API keys detected. Bot will run in READ-ONLY mode.")
    if not operator_ok:
        logger.warning("BITVAVO_OPERATOR_ID missing. Bot can NOT place orders (Bitvavo requires operatorId).")

    state = load_state()

    symbols = cfg["symbols"]
    quote = cfg["quote"]
    timeframe = cfg["timeframe"]
    candles_limit = int(cfg["logging"]["candles_limit"])
    sleep_s = int(cfg["logging"]["loop_sleep_seconds"])

    risk = cfg["risk"]
    max_spread = float(risk["max_spread_pct"])
    cooldown_min = float(risk["cooldown_minutes"])
    stake_eur = float(risk["fixed_stake_quote"])

    while True:
        try:
            # Balance
            eur_free = None
            bal = None
            if keys_ok:
                bal = safe_fetch_balance(ex, state, logger)
                eur_free = bal["EUR_free"] if bal else None

            if eur_free is not None:
                logger.info(f"Vrij EUR: {eur_free:.2f}")

            # Iterate markets
            for base in symbols:
                market = f"{base}/{quote}"

                # Load candles
                try:
                    df = fetch_ohlcv_df(ex, market, timeframe, candles_limit)
                    if len(df) < max(int(cfg["strategy"]["sma_slow"]), int(cfg["strategy"]["atr_len"])) + 5:
                        continue
                except Exception as e:
                    logger.warning(f"{market}: candle fetch failed: {e}")
                    continue

                # Spread filter
                try:
                    t = ex.fetch_ticker(market)
                    sp = spread_pct_from_ticker(t)
                except Exception:
                    sp = 999.0

                if sp > max_spread:
                    continue

                positions = state.get("positions", {})
                in_pos = market in positions

                # SELL logic only for bot-managed positions
                if in_pos:
                    pos = positions[market]
                    do_sell, reason, a, tp, sl = should_sell(df, pos, cfg)
                    if do_sell and keys_ok and operator_ok:
                        try:
                            amount_base = float(pos["amount_base"])
                            sell = place_market_sell(ex, market, amount_base)
                            sell_price = float(sell["price"])
                            sell_quote_gross = sell_price * amount_base
                            sell_fee = calc_fee_eur(cfg, sell_quote_gross)

                            buy_cost = float(pos["quote_cost_eur"])
                            buy_fee = float(pos["buy_fee_eur"])
                            net_pnl = sell_quote_gross - sell_fee - buy_cost - buy_fee

                            # holding time
                            entry_ts = datetime.fromisoformat(pos["entry_ts"])
                            hold_min = (now_utc() - entry_ts).total_seconds() / 60.0

                            # stats
                            state["pnl_eur"] = float(state.get("pnl_eur", 0.0)) + net_pnl
                            state["stats"]["closed"] = int(state["stats"]["closed"]) + 1
                            if net_pnl > 0:
                                state["stats"]["wins"] = int(state["stats"]["wins"]) + 1
                            else:
                                state["stats"]["losses"] = int(state["stats"]["losses"]) + 1

                            closed = int(state["stats"]["closed"])
                            wins = int(state["stats"]["wins"])
                            winrate = pct(wins, closed)

                            # cooldown
                            state.setdefault("cooldown", {})[market] = now_utc().timestamp() + cooldown_min * 60.0

                            append_trade({
                                "ts": now_utc().isoformat(),
                                "market": market,
                                "side": "VERKOPEN",
                                "reason": reason,
                                "price": round(sell_price, 10),
                                "amount_base": round(amount_base, 10),
                                "quote_gross_eur": round(sell_quote_gross, 2),
                                "fee_eur": round(sell_fee, 4),
                                "spread_pct": round(sp, 4),
                                "atr": round(a, 10),
                                "tp_price": round(tp, 10),
                                "sl_price": round(sl, 10),
                                "holding_time_min": round(hold_min, 2),
                                "netto_pnl_eur": round(net_pnl, 2),
                                "totaal_pnl_eur": round(float(state["pnl_eur"]), 2),
                                "winrate_pct": round(winrate, 2),
                            })

                            logger.info(
                                f"{market} verkocht. Netto PnL €{net_pnl:.2f}, totaal €{state['pnl_eur']:.2f}, "
                                f"trades {closed}, winrate {winrate:.2f}%, hold {hold_min:.1f} min"
                            )

                            # remove position
                            del positions[market]
                            state["positions"] = positions
                            save_state(state)

                        except Exception as e:
                            logger.error(f"{market}: sell failed: {e}")
                            logger.debug(traceback.format_exc())

                    continue  # do not buy same loop if in position

                # BUY logic
                if keys_ok and operator_ok and eur_free is not None:
                    ok, why = can_open_new_position(market, state, cfg, eur_free)
                    if not ok:
                        continue

                    buy_signal, sig_reason = should_buy_trend_v2(df, cfg)
                    if not buy_signal:
                        continue

                    # place buy
                    try:
                        buy = place_market_buy(ex, market, stake_eur)
                        buy_price = float(buy["price"])
                        amount_base = float(buy["amount_base"])
                        quote_gross = buy_price * amount_base
                        buy_fee = calc_fee_eur(cfg, quote_gross)

                        # store position (only bot-managed)
                        a = atr(df, int(cfg["strategy"]["atr_len"])).iloc[-1]
                        positions = state.get("positions", {})
                        positions[market] = {
                            "entry_price": buy_price,
                            "amount_base": amount_base,
                            "entry_ts": now_utc().isoformat(),
                            "quote_cost_eur": quote_gross,
                            "buy_fee_eur": buy_fee,
                            "atr": safe_float(a, 0.0),
                            "signal": sig_reason,
                        }
                        state["positions"] = positions

                        # cooldown to avoid immediate rebuys
                        state.setdefault("cooldown", {})[market] = now_utc().timestamp() + cooldown_min * 60.0

                        append_trade({
                            "ts": now_utc().isoformat(),
                            "market": market,
                            "side": "KOPEN",
                            "reason": sig_reason,
                            "price": round(buy_price, 10),
                            "amount_base": round(amount_base, 10),
                            "quote_gross_eur": round(quote_gross, 2),
                            "fee_eur": round(buy_fee, 4),
                            "spread_pct": round(sp, 4),
                            "atr": round(safe_float(a, 0.0), 10),
                            "tp_price": "",
                            "sl_price": "",
                            "holding_time_min": "",
                            "netto_pnl_eur": "",
                            "totaal_pnl_eur": round(float(state.get("pnl_eur", 0.0)), 2),
                            "winrate_pct": round(pct(int(state["stats"]["wins"]), max(1, int(state["stats"]["closed"]))), 2),
                        })

                        logger.info(
                            f"{market} gekocht voor netto €{quote_gross - buy_fee:.2f} @ {buy_price:.6g}, spread {sp:.3f}%"
                        )

                        save_state(state)

                    except Exception as e:
                        logger.error(f"{market}: buy failed: {e}")
                        logger.debug(traceback.format_exc())

            time.sleep(sleep_s)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            logger.debug(traceback.format_exc())
            time.sleep(max(5, sleep_s))

if __name__ == "__main__":
    main()
