#!/usr/bin/env python3
"""
Diamond Agent v2
- Controleert elke 6 uur saldo, verlies, markt
- Past stake automatisch aan op basis van saldo
- Pauzeert bot bij slechte markt of te veel verlies
- Hervat automatisch als situatie verbetert
- Email rapport 08:00 en 20:00
"""
import csv
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timezone
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

REPORT_HOURS      = [6, 18]  # 06:00 en 18:00 UTC = 08:00 en 20:00 Nederlandse tijd
ANALYZE_INTERVAL  = 6 * 3600
MAX_DAY_LOSS_PCT  = 1.5    # pauzeert bij meer dan 1.5% van totaal inleg als dagverlies
BTC_DROP_LIMIT    = 8.0    # pauzeert bij >8% BTC daling in 24u
BTC_RECOVER_PCT   = 4.0    # hervat als BTC 4% herstelt


def load_state():
    if not Path(STATE_FILE).exists():
        return {}
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def send_email(subject, body):
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
    """BTC prijsverandering in % over 24 uur."""
    try:
        candles = exchange.fetch_ohlcv("BTC/EUR", "1h", limit=25)
        if len(candles) < 2:
            return 0.0
        price_24h_ago = float(candles[0][4])
        price_now     = float(candles[-1][4])
        return ((price_now - price_24h_ago) / price_24h_ago) * 100
    except Exception:
        return 0.0


def get_day_pnl() -> float:
    """PnL van vandaag uit CSV."""
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
    trades = []
    if Path(TRADES_FILE).exists():
        trades = list(csv.DictReader(open(TRADES_FILE)))

    sells   = [t for t in trades if t.get("side") == "SELL"]
    total   = len(sells)
    wins    = sum(1 for t in sells if float(t.get("pnl") or 0) > 0)
    pnl     = sum(float(t.get("pnl") or 0) for t in sells)
    winrate = (wins / total * 100) if total else 0
    day_pnl = get_day_pnl()
    btc_chg = get_btc_change(exchange)

    # Saldo ophalen
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
        per_coin[sym]["trades"] += 1
        p = float(t.get("pnl") or 0)
        per_coin[sym]["pnl"] += p
        if p > 0:
            per_coin[sym]["wins"] += 1

    grids   = state.get("grids", {})
    paused  = state.get("paused", False)
    open_pos = sum(len(g.get("positions", {})) for g in grids.values())

    lines = [
        "=" * 50,
        f"  DIAMOND GRID BOT RAPPORT",
        f"  {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        "=" * 50,
        f"  Status        : {'GEPAUZEERD - ' + state.get('pause_reason','') if paused else 'ACTIEF'}",
        f"  Vrij saldo    : {free_eur:.2f} EUR",
        f"  BTC 24u       : {btc_chg:+.1f}%",
        f"  Dag PnL       : {day_pnl:+.2f} EUR",
        "",
        f"  Trades totaal : {total}",
        f"  Winst trades  : {wins}",
        f"  Verlies trades: {total - wins}",
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
    """Uitgebreid weekrapport met alle statistieken."""
    state  = load_state()
    trades = []
    if Path(TRADES_FILE).exists():
        trades = list(csv.DictReader(open(TRADES_FILE)))

    sells = [t for t in trades if t.get("side") == "SELL"]
    total = len(sells)
    wins  = sum(1 for t in sells if float(t.get("pnl") or 0) > 0)
    pnl   = sum(float(t.get("pnl") or 0) for t in sells)
    winrate = (wins / total * 100) if total else 0

    # Week statistieken
    from datetime import timedelta
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_sells = [t for t in sells if t.get("ts", "") >= week_ago]
    week_wins  = sum(1 for t in week_sells if float(t.get("pnl") or 0) > 0)
    week_pnl   = sum(float(t.get("pnl") or 0) for t in week_sells)
    week_wr    = (week_wins / len(week_sells) * 100) if week_sells else 0

    # Saldo
    try:
        bal = exchange.fetch_balance()
        free_eur = float((bal.get("free") or {}).get("EUR", 0))
    except Exception:
        free_eur = 0.0

    btc_chg = get_btc_change(exchange)
    grids   = state.get("grids", {})
    open_pos = sum(len(g.get("positions", {})) for g in grids.values())

    lines = [
        "=" * 50,
        f"  DIAMOND BOT WEEKRAPPORT",
        f"  {datetime.now().strftime('%d-%m-%Y')}",
        "=" * 50,
        "",
        "  DEZE WEEK:",
        f"  Trades        : {len(week_sells)}",
        f"  Winrate       : {week_wr:.1f}%",
        f"  PnL deze week : {week_pnl:+.2f} EUR",
        "",
        "  TOTAAL ALLTIME:",
        f"  Trades        : {total}",
        f"  Winrate       : {winrate:.1f}%",
        f"  Totaal PnL    : {pnl:+.2f} EUR",
        "",
        "  HUIDIGE STAND:",
        f"  Vrij saldo    : {free_eur:.2f} EUR",
        f"  Open posities : {open_pos}",
        f"  BTC 24u trend : {btc_chg:+.1f}%",
        f"  Bot status    : {'GEPAUZEERD' if state.get('paused') else 'ACTIEF'}",
        "",
        "  PER COIN ALLTIME:",
    ]

    per_coin = {}
    for t in sells:
        sym = t.get("market", "?")
        if sym not in per_coin:
            per_coin[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
        per_coin[sym]["trades"] += 1
        p = float(t.get("pnl") or 0)
        per_coin[sym]["pnl"] += p
        if p > 0:
            per_coin[sym]["wins"] += 1

    for sym, d in per_coin.items():
        wr = (d["wins"] / d["trades"] * 100) if d["trades"] else 0
        lines.append(f"    {sym:<12} trades={d['trades']} winrate={wr:.0f}% pnl={d['pnl']:+.2f} EUR")

    lines += [
        "",
        "  GRID RANGES:",
    ]
    for sym, g in grids.items():
        pos = g.get("positions", {})
        lines.append(f"    {sym:<12} {len(pos)} open | {g.get('low',0):.4f}-{g.get('high',0):.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)



def update_inleg_if_needed(exchange):
    """Check of er nieuwe stortingen zijn en update total_inleg."""
    state = load_state()
    try:
        # Haal transacties op van Bitvavo
        bal = exchange.fetch_balance()
        free = float((bal.get("free") or {}).get("EUR", 0))
        ingezet = sum(
            float(p.get("buy_cost", 0))
            for g in state.get("grids", {}).values()
            for p in g.get("positions", {}).values()
        )
        actueel_totaal = free + ingezet
        opgeslagen = float(state.get("total_inleg", 0))

        # Als actueel totaal meer dan €50 hoger is dan opgeslagen → nieuwe storting
        if actueel_totaal > opgeslagen + 50:
            verschil = actueel_totaal - opgeslagen
            state["total_inleg"] = actueel_totaal
            save_state(state)
            LOG.info("Nieuwe storting gedetecteerd: +%.2f EUR | totaal inleg: %.2f EUR", verschil, actueel_totaal)
            send_email(
                "💰 Diamond Bot - Storting gedetecteerd",
                f"Nieuwe storting van ~{verschil:.0f} EUR gedetecteerd. Totaal inleg: {actueel_totaal:.2f} EUR. Bot schaalt automatisch op."
            )
    except Exception as e:
        LOG.warning("Inleg check mislukt: %s", e)


def analyze_and_act(exchange):
    state   = load_state()
    btc_chg = get_btc_change(exchange)
    day_pnl = get_day_pnl()
    paused  = state.get("paused", False)

    # Check of we moeten pauzeren
    # Dagverlies limiet: 1.5% van totaal inleg
    total_inleg = float(state.get("total_inleg", 1500))
    max_day_loss = total_inleg * (MAX_DAY_LOSS_PCT / 100)

    if not paused:
        if day_pnl <= -max_day_loss:
            state["paused"] = True
            state["pause_reason"] = f"dagverlies_{day_pnl:.2f}_EUR_limiet_{max_day_loss:.2f}"
            state["pause_btc_price"] = None
            save_state(state)
            LOG.warning("BOT GEPAUZEERD | dagverlies=%.2f EUR", day_pnl)
            send_email(
                "⚠️ Diamond Bot GEPAUZEERD - Dagverlies",
                f"Bot gepauzeerd wegens dagverlies van {day_pnl:.2f} EUR.\nHervat automatisch morgen.\n\n{build_report(exchange)}"
            )

        elif btc_chg <= -BTC_DROP_LIMIT:
            try:
                ticker = exchange.fetch_ticker("BTC/EUR")
                btc_price = float(ticker.get("last") or 0)
            except Exception:
                btc_price = 0
            state["paused"] = True
            state["pause_reason"] = f"btc_daling_{btc_chg:.1f}pct"
            state["pause_btc_price"] = btc_price
            save_state(state)
            LOG.warning("BOT GEPAUZEERD | BTC daling=%.1f%%", btc_chg)
            send_email(
                "⚠️ Diamond Bot GEPAUZEERD - BTC Daling",
                f"Bot gepauzeerd wegens BTC daling van {btc_chg:.1f}%.\nHervat automatisch bij herstel.\n\n{build_report(exchange)}"
            )

    # Check of we kunnen hervatten
    elif paused:
        reason = state.get("pause_reason", "")

        # Dagverlies pauze: hervat volgende dag
        if "dagverlies" in reason:
            today = datetime.now().strftime("%Y-%m-%d")
            pause_date = state.get("pause_date", today)
            if today != pause_date:
                state["paused"] = False
                state["pause_reason"] = ""
                save_state(state)
                LOG.info("BOT HERVAT | nieuw dag")
                send_email("✅ Diamond Bot HERVAT", f"Bot hervat na dagverlies pauze.\n\n{build_report(exchange)}")

        # BTC daling pauze: hervat als BTC herstelt
        elif "btc_daling" in reason:
            pause_price = state.get("pause_btc_price", 0)
            try:
                ticker = exchange.fetch_ticker("BTC/EUR")
                btc_now = float(ticker.get("last") or 0)
            except Exception:
                btc_now = 0

            if pause_price > 0 and btc_now > 0:
                recovery = ((btc_now - pause_price) / pause_price) * 100
                if recovery >= BTC_RECOVER_PCT:
                    state["paused"] = False
                    state["pause_reason"] = ""
                    state["pause_btc_price"] = None
                    save_state(state)
                    LOG.info("BOT HERVAT | BTC herstel=%.1f%%", recovery)
                    send_email("✅ Diamond Bot HERVAT", f"Bot hervat na BTC herstel van {recovery:.1f}%.\n\n{build_report(exchange)}")

    LOG.info("Analyse klaar | btc_chg=%.1f%% | dag_pnl=%.2f EUR | paused=%s",
             btc_chg, day_pnl, state.get("paused", False))


def main():
    load_dotenv()
    exchange = ccxt.bitvavo({
        "apiKey":  os.getenv("BITVAVO_API_KEY", "").strip(),
        "secret":  os.getenv("BITVAVO_API_SECRET", "").strip(),
        "enableRateLimit": True,
    })
    exchange.load_markets()

    LOG.info("Diamond Agent v2 gestart")
    last_analyze   = 0.0
    last_report_hr = -1

    last_weekly_report = -1

    while True:
        now = datetime.now(timezone.utc)

        # Dagrapport
        if now.hour in REPORT_HOURS and now.hour != last_report_hr:
            report = build_report(exchange)
            send_email(f"Diamond Grid Bot Rapport {now.strftime('%d-%m-%Y %H:%M')}", report)
            last_report_hr = now.hour

        # Wekelijks rapport elke zondag om 09:00
        if now.weekday() == 6 and now.hour == 7 and now.day != last_weekly_report:  # 07:00 UTC = 09:00 NL
            report = build_weekly_report(exchange)
            send_email(f"📊 Diamond Bot WEEKRAPPORT {now.strftime('%d-%m-%Y')}", report)
            last_weekly_report = now.day
            LOG.info("Weekrapport verstuurd")

        # Analyse elke 6 uur
        if time.time() - last_analyze >= ANALYZE_INTERVAL:
            analyze_and_act(exchange)
            last_analyze = time.time()

        time.sleep(60)


if __name__ == "__main__":
    main()
