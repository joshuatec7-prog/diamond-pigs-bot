#!/usr/bin/env python3
"""
Diamond Agent v3
- Geen automatische storting detectie (handmatig via shell)
- Dagrapport 08:00 en 20:00 Nederlandse tijd (06:00 en 18:00 UTC)
- Weekrapport zondag 09:00 Nederlandse tijd (07:00 UTC)
- Pauzeert bij dagverlies > 1.5% van total_inleg
- Pauzeert bij BTC daling > 8%
- Hervat automatisch
"""

import csv
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

import ccxt
from dotenv import load_dotenv

LOG = logging.getLogger("agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

STATE_FILE  = "/opt/render/project/src/grid_state.json"
TRADES_FILE = "/opt/render/project/src/grid_transactions.csv"
GMAIL_USER  = "joshuatec7@gmail.com"
GMAIL_PASS  = os.getenv("GMAIL_APP_PASSWORD", "").strip()

REPORT_HOURS_UTC = [6, 18]   # 08:00 en 20:00 NL tijd
ANALYZE_INTERVAL = 6 * 3600  # elke 6 uur analyseren
BTC_DROP_LIMIT   = 8.0        # % daling om te pauzeren
BTC_RECOVER_PCT  = 4.0        # % herstel om te hervatten
MAX_DAY_LOSS_PCT = 1.5        # % van total_inleg als dagverlies limiet


def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return {}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def save_state(state: dict):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def send_email(subject: str, body: str):
    if not GMAIL_PASS:
        LOG.warning("Geen GMAIL_APP_PASSWORD")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = GMAIL_USER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASS)
            smtp.send_message(msg)
        LOG.info("Email verstuurd: %s", subject)
    except Exception as e:
        LOG.error("Email mislukt: %s", e)


def get_btc_change(exchange) -> float:
    try:
        candles = exchange.fetch_ohlcv("BTC/EUR", "1h", limit=25)
        if len(candles) < 2:
            return 0.0
        return ((float(candles[-1][4]) - float(candles[0][4])) / float(candles[0][4])) * 100
    except Exception:
        return 0.0


def get_day_pnl() -> float:
    if not Path(TRADES_FILE).exists():
        return 0.0
    today = datetime.now().strftime("%Y-%m-%d")
    total = 0.0
    try:
        for r in csv.DictReader(open(TRADES_FILE)):
            if r.get("ts", "").startswith(today) and r.get("side") == "SELL":
                total += float(r.get("pnl") or 0)
    except Exception:
        pass
    return total


def build_report(exchange) -> str:
    state  = load_state()
    trades = list(csv.DictReader(open(TRADES_FILE))) if Path(TRADES_FILE).exists() else []
    sells  = [t for t in trades if t.get("side") == "SELL"]
    total  = len(sells)
    wins   = sum(1 for t in sells if float(t.get("pnl") or 0) > 0)
    pnl    = sum(float(t.get("pnl") or 0) for t in sells)
    winrate = (wins / total * 100) if total else 0
    day_pnl = get_day_pnl()
    btc_chg = get_btc_change(exchange)

    try:
        bal = exchange.fetch_balance()
        free_eur = float((bal.get("free") or {}).get("EUR", 0))
    except Exception:
        free_eur = 0.0

    per_coin = {}
    for t in sells:
        sym = t.get("market", "?")
        if sym not in per_coin:
            per_coin[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
        p = float(t.get("pnl") or 0)
        per_coin[sym]["trades"] += 1
        per_coin[sym]["pnl"] += p
        if p > 0:
            per_coin[sym]["wins"] += 1

    grids    = state.get("grids", {})
    open_pos = sum(len(g.get("positions", {})) for g in grids.values())
    total_inleg = float(state.get("total_inleg", 0))

    lines = [
        "=" * 50,
        "  DIAMOND GRID BOT RAPPORT",
        f"  {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        "=" * 50,
        f"  Status        : {'GEPAUZEERD - ' + state.get('pause_reason','') if state.get('paused') else 'ACTIEF'}",
        f"  Totaal inleg  : {total_inleg:.2f} EUR",
        f"  Vrij saldo    : {free_eur:.2f} EUR",
        f"  BTC 24u       : {btc_chg:+.1f}%",
        f"  Dag PnL       : {day_pnl:+.2f} EUR",
        "",
        f"  Trades totaal : {total}",
        f"  Wins          : {wins}",
        f"  Verliezen     : {total - wins}",
        f"  Winrate       : {winrate:.1f}%",
        f"  Totaal PnL    : {pnl:+.2f} EUR",
        f"  Open posities : {open_pos}",
        "",
        "  PER COIN:",
    ]
    for sym, d in per_coin.items():
        wr = (d["wins"] / d["trades"] * 100) if d["trades"] else 0
        lines.append(f"    {sym:<12} trades={d['trades']} winrate={wr:.0f}% pnl={d['pnl']:+.2f} EUR")

    lines += ["", "  GRID RANGES:"]
    for sym, g in grids.items():
        pos = g.get("positions", {})
        lines.append(f"    {sym:<12} {len(pos)} open | {g.get('low',0):.4f}-{g.get('high',0):.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


def build_weekly_report(exchange) -> str:
    state  = load_state()
    trades = list(csv.DictReader(open(TRADES_FILE))) if Path(TRADES_FILE).exists() else []
    sells  = [t for t in trades if t.get("side") == "SELL"]
    total  = len(sells)
    wins   = sum(1 for t in sells if float(t.get("pnl") or 0) > 0)
    pnl    = sum(float(t.get("pnl") or 0) for t in sells)
    winrate = (wins / total * 100) if total else 0

    week_ago   = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_sells = [t for t in sells if t.get("ts", "") >= week_ago]
    week_wins  = sum(1 for t in week_sells if float(t.get("pnl") or 0) > 0)
    week_pnl   = sum(float(t.get("pnl") or 0) for t in week_sells)
    week_wr    = (week_wins / len(week_sells) * 100) if week_sells else 0

    try:
        bal = exchange.fetch_balance()
        free_eur = float((bal.get("free") or {}).get("EUR", 0))
    except Exception:
        free_eur = 0.0

    grids    = state.get("grids", {})
    open_pos = sum(len(g.get("positions", {})) for g in grids.values())
    total_inleg = float(state.get("total_inleg", 0))

    lines = [
        "=" * 50,
        "  DIAMOND BOT WEEKRAPPORT",
        f"  {datetime.now().strftime('%d-%m-%Y')}",
        "=" * 50,
        "",
        "  DEZE WEEK:",
        f"  Trades    : {len(week_sells)}",
        f"  Winrate   : {week_wr:.1f}%",
        f"  Week PnL  : {week_pnl:+.2f} EUR",
        "",
        "  ALLTIME:",
        f"  Trades    : {total}",
        f"  Winrate   : {winrate:.1f}%",
        f"  Totaal PnL: {pnl:+.2f} EUR",
        "",
        "  HUIDIGE STAND:",
        f"  Totaal inleg  : {total_inleg:.2f} EUR",
        f"  Vrij saldo    : {free_eur:.2f} EUR",
        f"  Open posities : {open_pos}",
        f"  Bot status    : {'GEPAUZEERD' if state.get('paused') else 'ACTIEF'}",
        "",
        "  PER COIN ALLTIME:",
    ]

    per_coin = {}
    for t in sells:
        sym = t.get("market", "?")
        if sym not in per_coin:
            per_coin[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
        p = float(t.get("pnl") or 0)
        per_coin[sym]["trades"] += 1
        per_coin[sym]["pnl"] += p
        if p > 0:
            per_coin[sym]["wins"] += 1

    for sym, d in per_coin.items():
        wr = (d["wins"] / d["trades"] * 100) if d["trades"] else 0
        lines.append(f"    {sym:<12} trades={d['trades']} winrate={wr:.0f}% pnl={d['pnl']:+.2f} EUR")

    lines += ["", "  GRID RANGES:"]
    for sym, g in grids.items():
        pos = g.get("positions", {})
        lines.append(f"    {sym:<12} {len(pos)} open | {g.get('low',0):.4f}-{g.get('high',0):.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


def analyze_and_act(exchange):
    state       = load_state()
    btc_chg     = get_btc_change(exchange)
    day_pnl     = get_day_pnl()
    paused      = state.get("paused", False)
    total_inleg = float(state.get("total_inleg", 1795))
    max_loss    = total_inleg * (MAX_DAY_LOSS_PCT / 100)

    if not paused:
        if day_pnl <= -max_loss:
            state["paused"]      = True
            state["pause_reason"] = f"dagverlies_{day_pnl:.2f}_EUR"
            state["pause_date"]  = datetime.now().strftime("%Y-%m-%d")
            save_state(state)
            LOG.warning("BOT GEPAUZEERD | dagverlies=%.2f EUR (limiet=%.2f)", day_pnl, max_loss)
            send_email("⚠️ Diamond Bot GEPAUZEERD - Dagverlies",
                       f"Bot gepauzeerd.\nDagverlies: {day_pnl:.2f} EUR (limiet: {max_loss:.2f} EUR)\n\n{build_report(exchange)}")

        elif btc_chg <= -BTC_DROP_LIMIT:
            try:
                btc_price = float(exchange.fetch_ticker("BTC/EUR").get("last") or 0)
            except Exception:
                btc_price = 0
            state["paused"]          = True
            state["pause_reason"]    = f"btc_daling_{btc_chg:.1f}pct"
            state["pause_btc_price"] = btc_price
            save_state(state)
            LOG.warning("BOT GEPAUZEERD | BTC daling=%.1f%%", btc_chg)
            send_email("⚠️ Diamond Bot GEPAUZEERD - BTC Daling",
                       f"Bot gepauzeerd.\nBTC daling: {btc_chg:.1f}%\n\n{build_report(exchange)}")

    else:
        reason = state.get("pause_reason", "")

        if "dagverlies" in reason:
            today      = datetime.now().strftime("%Y-%m-%d")
            pause_date = state.get("pause_date", today)
            if today != pause_date:
                state["paused"]       = False
                state["pause_reason"] = ""
                save_state(state)
                LOG.info("BOT HERVAT | nieuwe dag")
                send_email("✅ Diamond Bot HERVAT", f"Bot hervat na dagverlies pauze.\n\n{build_report(exchange)}")

        elif "btc_daling" in reason:
            pause_price = float(state.get("pause_btc_price") or 0)
            try:
                btc_now = float(exchange.fetch_ticker("BTC/EUR").get("last") or 0)
            except Exception:
                btc_now = 0
            if pause_price > 0 and btc_now > 0:
                recovery = ((btc_now - pause_price) / pause_price) * 100
                if recovery >= BTC_RECOVER_PCT:
                    state["paused"]          = False
                    state["pause_reason"]    = ""
                    state["pause_btc_price"] = None
                    save_state(state)
                    LOG.info("BOT HERVAT | BTC herstel=%.1f%%", recovery)
                    send_email("✅ Diamond Bot HERVAT",
                               f"Bot hervat na BTC herstel van {recovery:.1f}%.\n\n{build_report(exchange)}")

    LOG.info("Analyse klaar | btc=%.1f%% | dag_pnl=%.2f | paused=%s",
             btc_chg, day_pnl, state.get("paused", False))


def main():
    load_dotenv()
    exchange = ccxt.bitvavo({
        "apiKey":  os.getenv("BITVAVO_API_KEY", "").strip(),
        "secret":  os.getenv("BITVAVO_API_SECRET", "").strip(),
        "enableRateLimit": True,
    })
    exchange.load_markets()

    LOG.info("Diamond Agent v3 gestart")
    last_analyze    = 0.0
    last_report_hr  = -1
    last_weekly_day = -1

    while True:
        now = datetime.now(timezone.utc)

        # Dagrapport 08:00 en 20:00 NL = 06:00 en 18:00 UTC
        if now.hour in REPORT_HOURS_UTC and now.hour != last_report_hr:
            send_email(
                f"📊 Diamond Bot Rapport {now.strftime('%d-%m-%Y %H:%M')} UTC",
                build_report(exchange)
            )
            last_report_hr = now.hour

        # Weekrapport zondag 09:00 NL = 07:00 UTC
        if now.weekday() == 6 and now.hour == 7 and now.day != last_weekly_day:
            send_email(
                f"📊 Diamond Bot WEEKRAPPORT {now.strftime('%d-%m-%Y')}",
                build_weekly_report(exchange)
            )
            last_weekly_day = now.day
            LOG.info("Weekrapport verstuurd")

        # Analyse elke 6 uur
        if time.time() - last_analyze >= ANALYZE_INTERVAL:
            analyze_and_act(exchange)
            last_analyze = time.time()

        time.sleep(60)


if __name__ == "__main__":
    main()
