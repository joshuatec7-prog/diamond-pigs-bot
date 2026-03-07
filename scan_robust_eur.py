#!/usr/bin/env python3
# Robust EUR market scanner for Bitvavo via ccxt
#
# Matches your live bot settings:
# - Timeframe: 15m
# - SMA 20/60
# - RSI 14 >= 55
# - ATR 14
# - ATR TP 2.2
# - ATR SL 1.8
# - ATR filter >= 0.30%
# - Cooldown 45 minutes AFTER SELL only
# - Stake 30 EUR
# - Fee 0.25%
#
# Notes:
# - Scans all active /EUR markets
# - Uses robust OHLCV fetching with pagination + fallback
# - Filters on minimum liquidity and minimum trades per window
# - Ranks on consistency across two windows

import math
import time
import pandas as pd
import ccxt

# ===================== SETTINGS =====================
QUOTE = "EUR"
TIMEFRAME = "15m"

WINDOW_CANDLES = 700
TOTAL_CANDLES = WINDOW_CANDLES * 2

MIN_TRADES_PER_WINDOW = 20
SLEEP_BETWEEN_MARKETS = 0.15

MIN_AVG_QUOTE_VOL_EUR = 5_000

SMA_FAST = 20
SMA_SLOW = 60

RSI_LEN = 14
RSI_BUY_MIN = 55

ATR_LEN = 14
ATR_TP_MULT = 2.2
ATR_SL_MULT = 1.8
EXIT_ON_TREND_BREAK = True

USE_ATR_FILTER = True
MIN_ATR_PCT = 0.30

STAKE_QUOTE = 30.0
COOLDOWN_MINUTES = 45
TAKER_FEE_PCT = 0.25
SLIPPAGE_PCT = 0.00
# ====================================================

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

def timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe}")

def fetch_ohlcv_robust(ex, market: str, timeframe: str, total_candles: int):
    needed = total_candles + 100
    errors = []

    # 1) Try CCXT pagination
    try:
        calls = max(2, math.ceil(needed / 500))
        ohlcv = ex.fetch_ohlcv(
            market,
            timeframe=timeframe,
            limit=needed,
            params={
                "paginate": True,
                "paginationCalls": calls,
            },
        )
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
            if len(df) >= int(total_candles * 0.85):
                return df.tail(needed).reset_index(drop=True)
    except Exception as e:
        errors.append(f"paginate={e}")

    # 2) Fallback: time-based loop with since
    try:
        step_ms = timeframe_to_ms(timeframe)
        since = ex.milliseconds() - (needed * step_ms)
        rows = []
        max_loops = 10
        limit_per_call = min(1000, needed)

        for _ in range(max_loops):
            batch = ex.fetch_ohlcv(market, timeframe=timeframe, since=since, limit=limit_per_call)
            if not batch:
                break

            if rows:
                last_ts = rows[-1][0]
                batch = [row for row in batch if row[0] > last_ts]

            if not batch:
                break

            rows.extend(batch)
            since = batch[-1][0] + step_ms

            if len(rows) >= needed:
                break

            time.sleep(ex.rateLimit / 1000)

        if rows:
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
            if len(df) >= int(total_candles * 0.85):
                return df.tail(needed).reset_index(drop=True)
    except Exception as e:
        errors.append(f"since_loop={e}")

    raise RuntimeError("OHLCV fetch failed: " + " | ".join(errors))

def backtest_slice(df: pd.DataFrame, start_i: int, end_i: int):
    warmup = max(SMA_SLOW, RSI_LEN, ATR_LEN) + 2
    if (end_i - start_i) <= warmup:
        return None

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
    cooldown_until = 0  # minutes proxy, applies to re-entry only

    sells = 0
    wins = 0
    pnl_sum = 0.0
    pnls = []

    for i in range(start_i + warmup, end_i):
        p = float(df["close"].iloc[i])
        if p <= 0:
            continue

        t_min = i * tf_min

        sma_fast_v = float(df["sma_fast"].iloc[i])
        sma_slow_v = float(df["sma_slow"].iloc[i])
        trend_up = sma_fast_v > sma_slow_v
        trend_break = sma_fast_v < sma_slow_v

        rsi_v = float(df["rsi"].iloc[i]) if not math.isnan(float(df["rsi"].iloc[i])) else 50.0
        atr_v = float(df["atr"].iloc[i]) if not math.isnan(float(df["atr"].iloc[i])) else 0.0

        atr_pct = (atr_v / p * 100.0) if p > 0 else 0.0
        atr_ok = (atr_pct >= MIN_ATR_PCT) if USE_ATR_FILTER else True

        # SELL is ALWAYS allowed while in position
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

        # BUY is blocked by cooldown
        if t_min < cooldown_until:
            continue

        if trend_up and (rsi_v >= RSI_BUY_MIN) and atr_ok:
            buy_price = p * (1 + SLIPPAGE_PCT / 100.0)
            base_amt = STAKE_QUOTE / buy_price
            entry_price = buy_price
            entry_cost = STAKE_QUOTE
            entry_fee = entry_cost * (TAKER_FEE_PCT / 100.0)
            in_pos = True

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

    pairs = sorted([
        sym for sym, m in markets.items()
        if (m.get("active", True) is not False) and sym.endswith(f"/{QUOTE}")
    ])

    print(f"Scanning {len(pairs)} {QUOTE} markets... timeframe={TIMEFRAME}, total_candles={TOTAL_CANDLES}", flush=True)
    print(
        f"Filters: MIN_TRADES_PER_WINDOW={MIN_TRADES_PER_WINDOW}, "
        f"MIN_AVG_QUOTE_VOL_EUR={MIN_AVG_QUOTE_VOL_EUR}",
        flush=True,
    )

    rows = []

    for idx, market in enumerate(pairs, 1):
        if idx % 10 == 0:
            print(f"Progress {idx}/{len(pairs)}", flush=True)

        try:
            df = fetch_ohlcv_robust(ex, market, TIMEFRAME, TOTAL_CANDLES)

            if df.empty or len(df) < int(TOTAL_CANDLES * 0.85):
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            df["quote_vol"] = df["volume"] * df["close"]
            avg_qv = float(df["quote_vol"].tail(WINDOW_CANDLES).mean())
            if avg_qv < MIN_AVG_QUOTE_VOL_EUR:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            df["sma_fast"] = sma(df["close"], SMA_FAST)
            df["sma_slow"] = sma(df["close"], SMA_SLOW)
            df["rsi"] = rsi(df["close"], RSI_LEN)
            df["atr"] = atr(df, ATR_LEN)

            n = len(df)
            mid = n - WINDOW_CANDLES
            old_start, old_end = max(0, n - 2 * WINDOW_CANDLES), mid
            new_start, new_end = mid, n

            old = backtest_slice(df, old_start, old_end)
            new = backtest_slice(df, new_start, new_end)

            if not old or not new:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            if old["sells"] < MIN_TRADES_PER_WINDOW or new["sells"] < MIN_TRADES_PER_WINDOW:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            if old["pnl"] <= 0 or new["pnl"] <= 0:
                time.sleep(SLEEP_BETWEEN_MARKETS)
                continue

            min_wr = min(old["winrate"], new["winrate"])
            pf_min = min(old["profit_factor"], new["profit_factor"])
            pnl_total = old["pnl"] + new["pnl"]
            sells_total = old["sells"] + new["sells"]

            score = pnl_total * (min_wr / 100.0) * math.log(1 + sells_total) * max(1.0, pf_min)

            rows.append({
                "market": market,
                "avg_quote_vol_eur": avg_qv,
                "old_sells": old["sells"],
                "old_winrate": old["winrate"],
                "old_pnl": old["pnl"],
                "old_pf": old["profit_factor"],
                "new_sells": new["sells"],
                "new_winrate": new["winrate"],
                "new_pnl": new["pnl"],
                "new_pf": new["profit_factor"],
                "pnl_total": pnl_total,
                "score": score,
            })

        except Exception:
            pass

        time.sleep(SLEEP_BETWEEN_MARKETS)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No robust candidates found with current filters.", flush=True)
        print("Tip: lower MIN_TRADES_PER_WINDOW to 15 or MIN_AVG_QUOTE_VOL_EUR to 2000.", flush=True)
        return

    out = out.sort_values("score", ascending=False)

    print("\nTOP ROBUST (sorted by score)".ljust(60, "="), flush=True)
    cols = [
        "market", "score", "pnl_total", "avg_quote_vol_eur",
        "old_sells", "old_winrate", "old_pnl", "old_pf",
        "new_sells", "new_winrate", "new_pnl", "new_pf"
    ]
    print(out.head(20)[cols].to_string(index=False), flush=True)

    print("\nTOP by TOTAL PnL (robust-only)".ljust(60, "="), flush=True)
    out_pnl = out.sort_values("pnl_total", ascending=False)
    print(out_pnl.head(20)[cols].to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
