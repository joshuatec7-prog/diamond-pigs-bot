import math
import time
import pandas as pd
import ccxt

# ===================== SETTINGS (edit if you want) =====================
TIMEFRAME = "15m"
CANDLES_LIMIT = 500          # faster scan (was 900)
MIN_TRADES = 5               # show results sooner (was 10)
SLEEP_BETWEEN_MARKETS = 0.08 # be nice to rate limits but still quick

# Strategy (match your bot)
SMA_FAST = 20
SMA_SLOW = 60

RSI_LEN = 14
RSI_BUY_MIN = 55

ATR_LEN = 14
ATR_TP_MULT = 2.2
ATR_SL_MULT = 1.8
EXIT_ON_TREND_BREAK = True

USE_ATR_FILTER = True
MIN_ATR_PCT = 0.30           # ATR% of price (0.30 = 0.30%)

# Risk/fees approximation
STAKE_QUOTE = 15.0
COOLDOWN_MINUTES = 45
TAKER_FEE_PCT = 0.25         # fee per side (approx)
SLIPPAGE_PCT = 0.00          # set e.g. 0.05 for 0.05% slippage each side
# ======================================================================

def sma(series, length):
    return series.rolling(length).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(length).mean()
    roll_down = down.rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()

def backtest_market(df: pd.DataFrame):
    df = df.copy()
    df["sma_fast"] = sma(df["close"], SMA_FAST)
    df["sma_slow"] = sma(df["close"], SMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)

    warmup = max(SMA_SLOW, RSI_LEN, ATR_LEN) + 2
    if len(df) <= warmup:
        return None

    # timeframe minutes (assumes "15m")
    tf_min = 15
    if TIMEFRAME.endswith("m"):
        try:
            tf_min = int(TIMEFRAME[:-1])
        except Exception:
            tf_min = 15

    in_pos = False
    base_amt = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    entry_fee = 0.0
    cooldown_until = 0  # in minutes (proxy)

    sells = 0
    wins = 0
    pnl_sum = 0.0
    pnls = []

    for i in range(warmup, len(df)):
        p = float(df["close"].iloc[i])
        if p <= 0:
            continue

        t_min = i * tf_min
        if t_min < cooldown_until:
            continue

        sma_fast_v = float(df["sma_fast"].iloc[i])
        sma_slow_v = float(df["sma_slow"].iloc[i])
        trend_up = sma_fast_v > sma_slow_v
        trend_break = sma_fast_v < sma_slow_v

        rsi_v = float(df["rsi"].iloc[i]) if not math.isnan(float(df["rsi"].iloc[i])) else 50.0
        atr_v = float(df["atr"].iloc[i]) if not math.isnan(float(df["atr"].iloc[i])) else 0.0

        atr_pct = (atr_v / p * 100.0) if p > 0 else 0.0
        atr_ok = (atr_pct >= MIN_ATR_PCT) if USE_ATR_FILTER else True

        # SELL
        if in_pos:
            if atr_v <= 0:
                atr_v = max(entry_price * 0.002, 0.01)

            tp = entry_price + ATR_TP_MULT * atr_v
            sl = entry_price - ATR_SL_MULT * atr_v

            do_sell = (p >= tp) or (p <= sl) or (EXIT_ON_TREND_BREAK and trend_break)
            if do_sell:
                sell_price = p * (1 - SLIPPAGE_PCT / 100.0)
                proceeds = base_amt * sell_price
                sell_fee = proceeds * (TAKER_FEE_PCT / 100.0)

                net_pnl = (proceeds - entry_cost) - (entry_fee + sell_fee)
                pnl_sum += net_pnl
                pnls.append(net_pnl)
                sells += 1
                if net_pnl > 0:
                    wins += 1

                in_pos = False
                base_amt = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                entry_fee = 0.0
                cooldown_until = t_min + COOLDOWN_MINUTES
            continue

        # BUY
        if (not in_pos) and trend_up and (rsi_v >= RSI_BUY_MIN) and atr_ok:
            buy_price = p * (1 + SLIPPAGE_PCT / 100.0)
            base_amt = STAKE_QUOTE / buy_price
            entry_price = buy_price
            entry_cost = STAKE_QUOTE
            entry_fee = entry_cost * (TAKER_FEE_PCT / 100.0)
            in_pos = True
            cooldown_until = t_min + COOLDOWN_MINUTES

    if sells == 0:
        return {"sells": 0, "winrate": 0.0, "pnl": 0.0, "profit_factor": 0.0}

    sr = pd.Series(pnls)
    gross_win = float(sr[sr > 0].sum()) if (sr > 0).any() else 0.0
    gross_loss = float(-sr[sr < 0].sum()) if (sr < 0).any() else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    return {
        "sells": sells,
        "winrate": (wins / sells * 100.0),
        "pnl": pnl_sum,
        "profit_factor": profit_factor,
    }

def main():
    ex = ccxt.bitvavo({"enableRateLimit": True})
    markets = ex.load_markets()

    pairs = []
    for sym, m in markets.items():
        if not m.get("active", True):
            continue
        if sym.endswith("/EUR"):
            pairs.append(sym)

    pairs = sorted(pairs)
    print(f"Scanning {len(pairs)} EUR markets... timeframe={TIMEFRAME}, candles={CANDLES_LIMIT}", flush=True)

    rows = []
    for idx, market in enumerate(pairs, 1):
        if idx % 10 == 0:
            print(f"Progress {idx}/{len(pairs)}", flush=True)

        try:
            ohlcv = ex.fetch_ohlcv(market, timeframe=TIMEFRAME, limit=CANDLES_LIMIT)
            if not ohlcv or len(ohlcv) < 80:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            res = backtest_market(df)
            if not res:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            res["market"] = market
            rows.append(res)

        except Exception:
            pass

        time.sleep(SLEEP_BETWEEN_MARKETS)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No results (no markets or no candle data).", flush=True)
        return

    rank = out[out["sells"] >= MIN_TRADES].copy()
    rank = rank.sort_values(["winrate", "pnl"], ascending=[False, False])

    print("\nTOP by WINRATE (min trades =", MIN_TRADES, ")", flush=True)
    print(rank.head(20)[["market", "sells", "winrate", "pnl", "profit_factor"]].to_string(index=False), flush=True)

    best_pnl = out[out["sells"] >= MIN_TRADES].sort_values("pnl", ascending=False).head(20)
    print("\nTOP by PnL (min trades =", MIN_TRADES, ")", flush=True)
    print(best_pnl[["market", "sells", "winrate", "pnl", "profit_factor"]].to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
