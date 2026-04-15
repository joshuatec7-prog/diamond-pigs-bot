import os
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd
import yaml
from dotenv import load_dotenv


LOG = logging.getLogger("bitvavo_trend_bot")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "ja", "aan", "on", "waar"}:
        return True
    if s in {"0", "false", "no", "nee", "uit", "off", "onwaar"}:
        return False
    return default


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path_str: str) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config moet een YAML-dictionary zijn.")
    return data


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_state(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    if not p.exists():
        return {"positions": {}, "cooldown": {}, "pnl_quote": 0.0, "trades": 0, "wins": 0}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("positions", {})
    data.setdefault("cooldown", {})
    data.setdefault("pnl_quote", 0.0)
    data.setdefault("trades", 0)
    data.setdefault("wins", 0)
    return data


def save_state(path_str: str, state: Dict[str, Any]) -> None:
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_trade_csv(path_str: str, row: Dict[str, Any]) -> None:
    ensure_parent(path_str)
    cols = [
        "ts",
        "market",
        "side",
        "price",
        "base_amount",
        "quote_amount",
        "fees_quote",
        "spread_pct",
        "net_pnl_quote",
        "holding_time_min",
        "reason",
        "dry_run",
    ]
    exists = Path(path_str).exists()
    df = pd.DataFrame([row], columns=cols)
    df.to_csv(path_str, mode="a", index=False, header=not exists)


ALIASES = {
    "dry_run": ["risk.dry_run"],
    "log_level": ["logging.level"],
    "candles_limit": ["logging.candles_limit"],
    "loop_sleep_seconds": ["logging.loop_sleep_seconds"],
    "fixed_stake_quote": ["risk.fixed_stake_quote"],
    "max_open_positions": ["risk.max_open_positions", "trading.max_total_positions"],
    "max_spread_pct": ["risk.max_spread_pct"],
    "eur_reserve": ["risk.eur_reserve"],
    "risk.taker_fee_pct": ["fees.taker_fee_pct"],
    "strategy.sma_fast": ["signals.sma_fast"],
    "strategy.sma_slow": ["signals.sma_slow"],
    "strategy.rsi_len": ["signals.rsi_len"],
    "strategy.rsi_buy_min": ["signals.rsi_buy_min"],
    "strategy.rsi_buy_max": ["signals.rsi_buy_max"],
    "strategy.atr_len": ["signals.atr_len"],
    "strategy.atr_tp_mult": ["signals.atr_tp_mult"],
    "strategy.atr_sl_mult": ["signals.atr_sl_mult"],
    "strategy.min_atr_pct": ["signals.min_atr_pct"],
    "strategy.exit_on_trend_break": ["signals.exit_on_trend_break"],
}


def _get_path(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur



def get_cfg(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    value = _get_path(cfg, path, None)
    if value is not None:
        return value
    for alias in ALIASES.get(path, []):
        alias_value = _get_path(cfg, alias, None)
        if alias_value is not None:
            return alias_value
    return default



def normalize_symbol(symbol: str, quote: str) -> str:
    s = str(symbol).strip().upper()
    q = str(quote).strip().upper()
    if "/" in s:
        return s
    if "-" in s:
        parts = s.split("-", 1)
        return f"{parts[0]}/{parts[1]}"
    return f"{s}/{q}"


def utc_now_ts() -> float:
    return time.time()


def minutes_since(ts: float) -> float:
    return max(0.0, (utc_now_ts() - ts) / 60.0)


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def enrich_indicators(df: pd.DataFrame, sma_fast: int, sma_slow: int, rsi_len: int, atr_len: int) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(sma_slow).mean()
    out["rsi"] = compute_rsi(out["close"], rsi_len)
    out["atr"] = compute_atr(out, atr_len)
    out["atr_pct"] = (out["atr"] / out["close"]) * 100.0
    return out


class Bot:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.quote = str(get_cfg(cfg, "quote", "EUR")).upper()
        self.dry_run = to_bool(get_cfg(cfg, "dry_run", True), True)
        files = get_cfg(cfg, "files", {}) or {}
        self.state_file = str(files.get("state_file", "state.json"))
        self.trades_file = str(files.get("trades_file", "transactions.csv"))
        self.state = load_state(self.state_file)

        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY", "")
        api_secret = os.getenv("BITVAVO_API_SECRET", "")
        operator_id = os.getenv("BITVAVO_OPERATOR_ID", "").strip()

        if not operator_id:
            raise ValueError("BITVAVO_OPERATOR_ID ontbreekt in je environment variables.")

        self.operator_id = int(operator_id)
        self.exchange = ccxt.bitvavo(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"fetchMarkets": {"types": ["spot"]}},
            }
        )
        self.exchange.load_markets()

    def order_params(self) -> Dict[str, Any]:
        return {"operatorId": self.operator_id}

    def safe_fetch_balance(self) -> Dict[str, Any]:
        for i in range(3):
            try:
                return self.exchange.fetch_balance()
            except Exception as e:
                LOG.warning("fetch_balance poging %s mislukt: %s", i + 1, e)
                time.sleep(1.5 * (i + 1))
        raise RuntimeError("Kon saldo niet ophalen.")

    def free_quote_balance(self, quote: Optional[str] = None) -> float:
        quote = str(quote or self.quote).upper()
        bal = self.safe_fetch_balance()
        free = bal.get("free", {}) or {}
        total = bal.get("total", {}) or {}
        if quote in free and free[quote] is not None:
            return float(free[quote])
        if quote in total and total[quote] is not None:
            return float(total[quote])
        return 0.0

    def fetch_ohlcv_df(self, symbol: str) -> pd.DataFrame:
        timeframe = str(get_cfg(self.cfg, "timeframe", "15m"))
        limit = int(get_cfg(self.cfg, "candles_limit", 300))
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        if df.empty:
            raise ValueError(f"Geen candles voor {symbol}")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    def market_min_notional(self, symbol: str) -> float:
        m = self.exchange.market(symbol)
        limit_cost = (((m.get("limits") or {}).get("cost") or {}).get("min"))
        if limit_cost:
            return float(limit_cost)
        info = m.get("info") or {}
        raw = info.get("minOrderInQuoteAsset")
        return to_float(raw, 5.0)

    def amount_to_precision_safe(self, symbol: str, amount: float) -> float:
        amt = self.exchange.amount_to_precision(symbol, amount)
        return float(amt)

    def estimate_spread_pct(self, ticker: Dict[str, Any]) -> float:
        bid = to_float(ticker.get("bid"), 0.0)
        ask = to_float(ticker.get("ask"), 0.0)
        if bid <= 0 or ask <= 0:
            return 999.0
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 100.0

    def scanned_symbols(self) -> List[str]:
        manual = get_cfg(self.cfg, "symbols", []) or []
        if manual:
            return [normalize_symbol(str(s), self.quote) for s in manual]

        top_n = int(get_cfg(self.cfg, "scanner.top_n_markets", 30))
        min_quote_volume = to_float(get_cfg(self.cfg, "scanner.min_quote_volume", 250000), 250000.0)
        exclude_bases = {str(x).upper() for x in (get_cfg(self.cfg, "scanner.exclude_bases", ["EUR", "USDT", "USDC"]) or [])}
        max_spread_pct = to_float(get_cfg(self.cfg, "max_spread_pct", 0.35), 0.35)

        candidates = []
        tickers = self.exchange.fetch_tickers()
        for symbol, t in tickers.items():
            try:
                market = self.exchange.market(symbol)
            except Exception:
                continue
            if not market.get("spot", True):
                continue
            if market.get("quote") != self.quote:
                continue
            base = str(market.get("base", "")).upper()
            if base in exclude_bases:
                continue
            qv = t.get("quoteVolume")
            if qv is None:
                info = t.get("info") or {}
                qv = info.get("quoteVolume") or info.get("volumeQuote")
            qv = to_float(qv, 0.0)
            spread_pct = self.estimate_spread_pct(t)
            if qv < min_quote_volume:
                continue
            if spread_pct > max_spread_pct:
                continue
            last = to_float(t.get("last"), 0.0)
            bid = to_float(t.get("bid"), 0.0)
            ask = to_float(t.get("ask"), 0.0)
            if min(last, bid, ask) <= 0:
                continue
            candidates.append((symbol, qv, spread_pct))
        candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
        return [x[0] for x in candidates[:top_n]]

    def symbol_in_cooldown(self, symbol: str) -> bool:
        cd = self.state.get("cooldown", {}).get(symbol)
        if not cd:
            return False
        cooldown_minutes = to_float(get_cfg(self.cfg, "risk.cooldown_minutes", 45), 45.0)
        return minutes_since(float(cd)) < cooldown_minutes

    def open_positions_count(self) -> int:
        return len(self.state.get("positions", {}))

    def bot_invested_quote(self) -> float:
        total = 0.0
        for pos in self.state.get("positions", {}).values():
            total += to_float(pos.get("quote_amount"), 0.0)
        return total

    def buy_budget_available(self) -> float:
        eur_reserve = to_float(get_cfg(self.cfg, "eur_reserve", 50), 50.0)
        capital_limit = to_float(get_cfg(self.cfg, "capital_limit_quote", 150), 150.0)
        free_quote = self.free_quote_balance(self.quote)
        room_from_balance = max(0.0, free_quote - eur_reserve)
        room_from_cap = max(0.0, capital_limit - self.bot_invested_quote())
        return min(room_from_balance, room_from_cap)

    def entry_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        df = self.fetch_ohlcv_df(symbol)
        df = enrich_indicators(
            df,
            sma_fast=int(get_cfg(self.cfg, "strategy.sma_fast", 20)),
            sma_slow=int(get_cfg(self.cfg, "strategy.sma_slow", 60)),
            rsi_len=int(get_cfg(self.cfg, "strategy.rsi_len", 14)),
            atr_len=int(get_cfg(self.cfg, "strategy.atr_len", 14)),
        )
        if len(df) < max(int(get_cfg(self.cfg, "strategy.sma_slow", 60)) + 2, 80):
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        fast_now = to_float(last["sma_fast"], 0.0)
        slow_now = to_float(last["sma_slow"], 0.0)
        fast_prev = to_float(prev["sma_fast"], 0.0)
        slow_prev = to_float(prev["sma_slow"], 0.0)
        close_now = to_float(last["close"], 0.0)
        rsi_now = to_float(last["rsi"], 50.0)
        atr_now = to_float(last["atr"], 0.0)
        atr_pct = to_float(last["atr_pct"], 0.0)

        rsi_min = to_float(get_cfg(self.cfg, "strategy.rsi_buy_min", 55), 55.0)
        rsi_max = to_float(get_cfg(self.cfg, "strategy.rsi_buy_max", 72), 72.0)
        min_atr_pct = to_float(get_cfg(self.cfg, "strategy.min_atr_pct", 0.25), 0.25)

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        trend_ok = fast_now > slow_now and close_now > fast_now
        momentum_ok = rsi_min <= rsi_now <= rsi_max
        volatility_ok = atr_pct >= min_atr_pct

        if cross_up and trend_ok and momentum_ok and volatility_ok and atr_now > 0:
            tp_mult = to_float(get_cfg(self.cfg, "strategy.atr_tp_mult", 2.5), 2.5)
            sl_mult = to_float(get_cfg(self.cfg, "strategy.atr_sl_mult", 1.4), 1.4)
            return {
                "close": close_now,
                "atr": atr_now,
                "rsi": rsi_now,
                "atr_pct": atr_pct,
                "stop_loss": close_now - atr_now * sl_mult,
                "take_profit": close_now + atr_now * tp_mult,
            }
        return None

    def exit_signal(self, symbol: str, position: Dict[str, Any]) -> Optional[str]:
        df = self.fetch_ohlcv_df(symbol)
        df = enrich_indicators(
            df,
            sma_fast=int(get_cfg(self.cfg, "strategy.sma_fast", 20)),
            sma_slow=int(get_cfg(self.cfg, "strategy.sma_slow", 60)),
            rsi_len=int(get_cfg(self.cfg, "strategy.rsi_len", 14)),
            atr_len=int(get_cfg(self.cfg, "strategy.atr_len", 14)),
        )
        last = df.iloc[-1]
        price = to_float(last["close"], 0.0)
        fast = to_float(last["sma_fast"], 0.0)
        slow = to_float(last["sma_slow"], 0.0)
        stop_loss = to_float(position.get("stop_loss"), 0.0)
        take_profit = to_float(position.get("take_profit"), 0.0)
        exit_on_trend_break = to_bool(get_cfg(self.cfg, "strategy.exit_on_trend_break", True), True)

        highest = max(to_float(position.get("highest_price"), 0.0), price)
        position["highest_price"] = highest

        trailing_enabled = to_bool(get_cfg(self.cfg, "strategy.trailing_enabled", True), True)
        trailing_atr_mult = to_float(get_cfg(self.cfg, "strategy.trailing_atr_mult", 1.2), 1.2)
        atr = to_float(last["atr"], 0.0)
        if trailing_enabled and atr > 0 and highest > 0:
            trailing_stop = highest - atr * trailing_atr_mult
            if trailing_stop > stop_loss:
                position["stop_loss"] = trailing_stop
                stop_loss = trailing_stop

        entry_price = to_float(position.get("entry_price"), 0.0)

        if stop_loss > 0 and price <= stop_loss:
            if stop_loss > entry_price and highest > entry_price:
                return "trailing_stop"
            return "stop_loss"
        if take_profit > 0 and price >= take_profit:
            return "take_profit"
        if exit_on_trend_break and fast < slow:
            return "trend_break"
        return None

    def place_market_buy(self, symbol: str, stake_quote: float) -> Dict[str, Any]:
        ticker = self.get_ticker(symbol)
        ask = to_float(ticker.get("ask"), 0.0)
        if ask <= 0:
            raise ValueError(f"Geen geldige ask voor {symbol}")

        amount = self.amount_to_precision_safe(symbol, stake_quote / ask)
        min_notional = self.market_min_notional(symbol)
        est_quote = amount * ask
        if est_quote < min_notional:
            raise ValueError(f"{symbol} te klein voor minimale orderwaarde. Nodig: {min_notional:.2f} {self.quote}")

        if self.dry_run:
            return {
                "id": f"drybuy-{int(time.time())}",
                "symbol": symbol,
                "price": ask,
                "amount": amount,
                "cost": est_quote,
                "fee": {"cost": est_quote * (to_float(get_cfg(self.cfg, "risk.taker_fee_pct", 0.25), 0.25) / 100.0), "currency": self.quote},
            }

        return self.exchange.create_order(symbol, "market", "buy", amount, None, {"operatorId": self.operator_id})

    def place_market_sell(self, symbol: str, amount: float) -> Dict[str, Any]:
        amount = self.amount_to_precision_safe(symbol, amount)
        if amount <= 0:
            raise ValueError("Verkoopamount is 0.")

        ticker = self.get_ticker(symbol)
        bid = to_float(ticker.get("bid"), 0.0)
        if bid <= 0:
            raise ValueError(f"Geen geldige bid voor {symbol}")

        if self.dry_run:
            est_quote = amount * bid
            return {
                "id": f"drysell-{int(time.time())}",
                "symbol": symbol,
                "price": bid,
                "amount": amount,
                "cost": est_quote,
                "fee": {"cost": est_quote * (to_float(get_cfg(self.cfg, "risk.taker_fee_pct", 0.25), 0.25) / 100.0), "currency": "EUR"},
            }

        return self.exchange.create_order(symbol, "market", "sell", amount, None, {"operatorId": self.operator_id})

    def order_fee_quote(self, order: Dict[str, Any], fallback_quote_amount: float) -> float:
        fee = order.get("fee")
        if isinstance(fee, dict) and fee.get("cost") is not None:
            return to_float(fee.get("cost"), 0.0)
        fees = order.get("fees") or []
        if isinstance(fees, list):
            total = 0.0
            for item in fees:
                total += to_float((item or {}).get("cost"), 0.0)
            if total > 0:
                return total
        taker_fee_pct = to_float(get_cfg(self.cfg, "risk.taker_fee_pct", 0.25), 0.25)
        return fallback_quote_amount * (taker_fee_pct / 100.0)

    def estimated_exit_pnl_quote(self, symbol: str, position: Dict[str, Any], bid_price: Optional[float] = None) -> float:
        if bid_price is None:
            ticker = self.get_ticker(symbol)
            bid_price = to_float(ticker.get("bid"), 0.0)
        amount = to_float(position.get("amount"), 0.0)
        gross_quote = amount * max(bid_price or 0.0, 0.0)
        taker_fee_pct = to_float(get_cfg(self.cfg, "risk.taker_fee_pct", 0.25), 0.25)
        est_sell_fee = gross_quote * (taker_fee_pct / 100.0)
        entry_quote = to_float(position.get("quote_amount"), 0.0)
        fee_buy_quote = to_float(position.get("fees_buy_quote"), 0.0)
        return gross_quote - est_sell_fee - entry_quote - fee_buy_quote

    def sell_allowed_by_profit(self, symbol: str, position: Dict[str, Any], reason: str) -> bool:
        if reason == "stop_loss":
            return True

        min_profit_eur = to_float(get_cfg(self.cfg, "risk.min_profit_eur", 0.0), 0.0)
        if min_profit_eur <= 0:
            return True

        ticker = self.get_ticker(symbol)
        bid = to_float(ticker.get("bid"), 0.0)
        if bid <= 0:
            return False

        est_pnl = self.estimated_exit_pnl_quote(symbol, position, bid)
        if est_pnl >= min_profit_eur:
            return True

        LOG.info(
            "HOLD %s | reason=%s | est_pnl=%.4f %s < min_profit=%.4f %s",
            symbol,
            reason,
            est_pnl,
            self.quote,
            min_profit_eur,
            self.quote,
        )
        return False

    def try_buy_symbol(self, symbol: str) -> None:
        if symbol in self.state["positions"]:
            return
        if self.symbol_in_cooldown(symbol):
            return
        if self.open_positions_count() >= int(get_cfg(self.cfg, "max_open_positions", 5)):
            return

        signal = self.entry_signal(symbol)
        if not signal:
            return

        fixed_stake = to_float(get_cfg(self.cfg, "fixed_stake_quote", 30), 30.0)
        budget = self.buy_budget_available()
        stake = min(fixed_stake, budget)
        if stake <= 0:
            LOG.info("Geen koopruimte meer. Budget beschikbaar: %.2f %s", budget, self.quote)
            return

        try:
            ticker = self.get_ticker(symbol)
            spread_pct = self.estimate_spread_pct(ticker)
            max_spread_pct = to_float(get_cfg(self.cfg, "max_spread_pct", 0.35), 0.35)
            if spread_pct > max_spread_pct:
                LOG.info("SKIP BUY %s spread %.3f%% > %.3f%%", symbol, spread_pct, max_spread_pct)
                return

            order = self.place_market_buy(symbol, stake)
            price = to_float(order.get("average") or order.get("price") or signal["close"], signal["close"])
            amount = to_float(order.get("filled") or order.get("amount"), 0.0)
            if amount <= 0:
                amount = self.amount_to_precision_safe(symbol, stake / max(price, 1e-12))
            quote_amount = to_float(order.get("cost"), amount * price)
            fee_quote = self.order_fee_quote(order, quote_amount)

            self.state["positions"][symbol] = {
                "opened_by_bot": True,
                "opened_at": utc_now_ts(),
                "entry_price": price,
                "amount": amount,
                "quote_amount": quote_amount,
                "fees_buy_quote": fee_quote,
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "highest_price": price,
            }
            save_state(self.state_file, self.state)

            append_trade_csv(
                self.trades_file,
                {
                    "ts": now_iso(),
                    "market": symbol,
                    "side": "BUY",
                    "price": round(price, 12),
                    "base_amount": amount,
                    "quote_amount": round(quote_amount, 8),
                    "fees_quote": round(fee_quote, 8),
                    "spread_pct": round(spread_pct, 6),
                    "net_pnl_quote": "",
                    "holding_time_min": "",
                    "reason": "entry_signal",
                    "dry_run": self.dry_run,
                },
            )

            LOG.info(
                "BUY %s | price=%.8f amount=%s quote=%.2f rsi=%.2f atr%%=%.3f stop=%.8f tp=%.8f dry=%s",
                symbol,
                price,
                amount,
                quote_amount,
                signal["rsi"],
                signal["atr_pct"],
                signal["stop_loss"],
                signal["take_profit"],
                self.dry_run,
            )
        except Exception as e:
            LOG.exception("BUY mislukt voor %s: %s", symbol, e)

    def try_sell_symbol(self, symbol: str, position: Dict[str, Any], reason: str) -> None:
        if not to_bool(position.get("opened_by_bot"), False):
            return
        try:
            amount = to_float(position.get("amount"), 0.0)
            order = self.place_market_sell(symbol, amount)
            price = to_float(order.get("average") or order.get("price"), 0.0)
            filled_amount = to_float(order.get("filled") or order.get("amount"), amount)
            quote_amount = to_float(order.get("cost"), filled_amount * price)
            fee_sell_quote = self.order_fee_quote(order, quote_amount)

            entry_quote = to_float(position.get("quote_amount"), 0.0)
            fee_buy_quote = to_float(position.get("fees_buy_quote"), 0.0)
            net_pnl_quote = quote_amount - fee_sell_quote - entry_quote - fee_buy_quote
            holding_time_min = minutes_since(float(position.get("opened_at", utc_now_ts())))

            self.state["pnl_quote"] = to_float(self.state.get("pnl_quote"), 0.0) + net_pnl_quote
            self.state["trades"] = int(self.state.get("trades", 0)) + 1
            if net_pnl_quote > 0:
                self.state["wins"] = int(self.state.get("wins", 0)) + 1

            self.state["positions"].pop(symbol, None)
            self.state["cooldown"][symbol] = utc_now_ts()
            save_state(self.state_file, self.state)

            ticker = self.get_ticker(symbol)
            spread_pct = self.estimate_spread_pct(ticker)

            append_trade_csv(
                self.trades_file,
                {
                    "ts": now_iso(),
                    "market": symbol,
                    "side": "SELL",
                    "price": round(price, 12),
                    "base_amount": filled_amount,
                    "quote_amount": round(quote_amount, 8),
                    "fees_quote": round(fee_sell_quote, 8),
                    "spread_pct": round(spread_pct, 6),
                    "net_pnl_quote": round(net_pnl_quote, 8),
                    "holding_time_min": round(holding_time_min, 2),
                    "reason": reason,
                    "dry_run": self.dry_run,
                },
            )

            LOG.info(
                "SELL %s | price=%.8f amount=%s quote=%.2f pnl=%.4f reason=%s dry=%s",
                symbol,
                price,
                filled_amount,
                quote_amount,
                net_pnl_quote,
                reason,
                self.dry_run,
            )
        except Exception as e:
            LOG.exception("SELL mislukt voor %s: %s", symbol, e)

    def manage_open_positions(self) -> None:
        positions = list((self.state.get("positions") or {}).items())
        for symbol, position in positions:
            try:
                reason = self.exit_signal(symbol, position)
                save_state(self.state_file, self.state)
                if reason and self.sell_allowed_by_profit(symbol, position, reason):
                    self.try_sell_symbol(symbol, position, reason)
            except Exception as e:
                LOG.exception("Positiebeheer mislukt voor %s: %s", symbol, e)

    def print_status(self, symbols: List[str]) -> None:
        pnl = to_float(self.state.get("pnl_quote"), 0.0)
        trades = int(self.state.get("trades", 0))
        wins = int(self.state.get("wins", 0))
        winrate = (wins / trades * 100.0) if trades > 0 else 0.0
        LOG.info(
            "STATUS | dry=%s | symbols=%s | open=%s | invested=%.2f | pnl=%.2f | trades=%s | winrate=%.1f%%",
            self.dry_run,
            len(symbols),
            self.open_positions_count(),
            self.bot_invested_quote(),
            pnl,
            trades,
            winrate,
        )

    def run_once(self) -> None:
        self.manage_open_positions()
        symbols = self.scanned_symbols()
        self.print_status(symbols)
        for symbol in symbols:
            if self.open_positions_count() >= int(get_cfg(self.cfg, "max_open_positions", 5)):
                break
            self.try_buy_symbol(symbol)

    def run_forever(self) -> None:
        sleep_s = int(get_cfg(self.cfg, "loop_sleep_seconds", 60))
        while True:
            try:
                self.run_once()
            except Exception as e:
                LOG.exception("Hoofdloop fout: %s", e)
            time.sleep(sleep_s)


def main() -> None:
    cfg_path = os.getenv("CFG_FILE", "config.yaml")
    cfg = load_yaml(cfg_path)
    setup_logging(str(get_cfg(cfg, "log_level", "INFO")))
    bot = Bot(cfg)
    LOG.info("Bitvavo trend bot gestart | dry_run=%s", bot.dry_run)
    bot.run_forever()


if __name__ == "__main__":
    main()
