#!/usr/bin/env python3
"""
Diamond Bot Agent
- Analyseert elke 6 uur de trades en past config automatisch aan
- Stuurt elke dag om 08:00 een dagrapport via email
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

import yaml

LOG = logging.getLogger("diamond_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

CONFIG_FILE = os.getenv("CFG_FILE", "/opt/render/project/src/config.yaml")
TRADES_FILE = "/opt/render/project/src/transactions.csv"
STATE_FILE  = "/opt/render/project/src/state.json"

GMAIL_USER     = "joshuatec7@gmail.com"
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()
REPORT_HOUR    = 8  # 08:00 UTC

ANALYZE_INTERVAL_SEC = 6 * 3600  # elke 6 uur

# Grenzen waarbinnen de agent mag aanpassen
LIMITS = {
    "rsi_buy_min":      (50, 68),
    "rsi_buy_max":      (68, 82),
    "atr_sl_mult":      (1.2, 3.0),
    "hard_stop_loss_pct": (5.0, 15.0),
    "min_atr_pct":      (0.20, 1.0),
    "atr_tp_mult":      (1.5, 4.0),
}


def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_trades():
    if not Path(TRADES_FILE).exists():
        return []
    trades = []
    with open(TRADES_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("side", "").upper() == "SELL":
                trades.append({
                    "symbol": r.get("market", ""),
                    "pnl":    float(r.get("net_pnl_quote") or 0),
                    "reason": r.get("reason", ""),
                })
    return trades


def load_state():
    if not Path(STATE_FILE).exists():
        return {}
    return json.load(open(STATE_FILE))


def analyze(trades):
    """Geeft aanbevelingen terug op basis van trade analyse."""
    if len(trades) < 5:
        return {}

    stop_losses  = [t for t in trades if "stop_loss" in t["reason"]]
    take_profits = [t for t in trades if "take_profit" in t["reason"] or "trailing" in t["reason"]]
    total        = len(trades)
    sl_rate      = len(stop_losses) / total
    tp_rate      = len(take_profits) / total
    winrate      = sum(1 for t in trades if t["pnl"] > 0) / total
    avg_loss     = sum(t["pnl"] for t in stop_losses) / len(stop_losses) if stop_losses else 0
    avg_win      = sum(t["pnl"] for t in take_profits) / len(take_profits) if take_profits else 0

    LOG.info(
        "Analyse: %s trades | winrate=%.1f%% | sl_rate=%.1f%% | tp_rate=%.1f%% | avg_loss=%.2f | avg_win=%.2f",
        total, winrate * 100, sl_rate * 100, tp_rate * 100, avg_loss, avg_win,
    )

    cfg  = load_config()
    sigs = cfg.get("signals", {})
    changes = {}

    # Te veel stop losses → stop verder weg zetten
    if sl_rate > 0.50:
        stap = 0.4 if sl_rate > 0.70 else 0.2
        new_sl = min(float(sigs.get("atr_sl_mult", 1.2)) + stap, LIMITS["atr_sl_mult"][1])
        new_hard = min(float(sigs.get("hard_stop_loss_pct", 7.0)) + 2.0, LIMITS["hard_stop_loss_pct"][1])
        changes["atr_sl_mult"] = round(new_sl, 2)
        changes["hard_stop_loss_pct"] = round(new_hard, 1)
        LOG.info("Veel stop losses (%.0f%%) → atr_sl_mult=%.2f hard_stop=%.1f%%", sl_rate*100, new_sl, new_hard)

    # Winrate te laag → instapfilters flink aanscherpen
    if winrate < 0.35:
        stap_rsi = 3 if winrate < 0.25 else 2
        stap_atr = 0.10 if winrate < 0.25 else 0.05
        new_rsi_min = min(float(sigs.get("rsi_buy_min", 58)) + stap_rsi, LIMITS["rsi_buy_min"][1])
        new_atr     = min(float(sigs.get("min_atr_pct", 0.30)) + stap_atr, LIMITS["min_atr_pct"][1])
        changes["rsi_buy_min"] = round(new_rsi_min, 1)
        changes["min_atr_pct"] = round(new_atr, 2)
        LOG.info("Lage winrate (%.0f%%) → rsi_buy_min=%.1f min_atr_pct=%.2f", winrate*100, new_rsi_min, new_atr)

    # Winrate goed → filters iets versoepelen voor meer trades
    if winrate > 0.55 and total >= 10:
        new_rsi_min = max(float(sigs.get("rsi_buy_min", 58)) - 1, LIMITS["rsi_buy_min"][0])
        changes["rsi_buy_min"] = round(new_rsi_min, 1)
        LOG.info("Goede winrate (%.0f%%) → rsi_buy_min iets lager: %.1f", winrate*100, new_rsi_min)

    # Gemiddelde winst klein → take profit hoger zetten
    if avg_win < 1.0 and tp_rate > 0.2:
        new_tp = min(float(sigs.get("atr_tp_mult", 2.6)) + 0.3, LIMITS["atr_tp_mult"][1])
        changes["atr_tp_mult"] = round(new_tp, 2)
        LOG.info("Kleine winsten (%.2f EUR) → atr_tp_mult=%.2f", avg_win, new_tp)

    # Verlies/winst ratio slecht → cooldown verhogen
    if winrate < 0.30 and total >= 10:
        risk = load_config().get("risk", {})
        new_cooldown = min(int(risk.get("cooldown_minutes", 90)) + 30, 240)
        cfg["risk"]["cooldown_minutes"] = new_cooldown
        changes["cooldown_minutes"] = new_cooldown
        LOG.info("Slechte ratio → cooldown=%s min", new_cooldown)

    return changes


def apply_changes(changes):
    if not changes:
        return
    cfg = load_config()
    for key, val in changes.items():
        cfg["signals"][key] = val
    save_config(cfg)
    LOG.info("Config aangepast: %s", changes)


def send_email(subject, body):
    if not GMAIL_PASSWORD:
        LOG.warning("Geen GMAIL_APP_PASSWORD gevonden, email niet verstuurd")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = GMAIL_USER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASSWORD)
            smtp.send_message(msg)
        LOG.info("Email verstuurd: %s", subject)
    except Exception as e:
        LOG.error("Email versturen mislukt: %s", e)


def build_report():
    trades = load_trades()
    state  = load_state()
    cfg    = load_config()

    total   = len(trades)
    wins    = sum(1 for t in trades if t["pnl"] > 0)
    pnl     = sum(t["pnl"] for t in trades)
    winrate = (wins / total * 100) if total else 0
    open_p  = state.get("positions", {})
    sigs    = cfg.get("signals", {})

    lines = [
        "=" * 50,
        f"  DIAMOND BOT DAGRAPPORT - {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        "=" * 50,
        f"  Trades totaal  : {total}",
        f"  Winst trades   : {wins}",
        f"  Verlies trades : {total - wins}",
        f"  Winrate        : {winrate:.1f}%",
        f"  Totaal PnL     : {pnl:+.2f} EUR",
        "",
        f"  Open posities  : {len(open_p)}",
    ]
    for sym, pos in open_p.items():
        lines.append(f"    {sym}: {float(pos.get('quote_amount',0)):.2f} EUR")

    lines += [
        "",
        "  HUIDIGE INSTELLINGEN:",
        f"    RSI min/max    : {sigs.get('rsi_buy_min')} / {sigs.get('rsi_buy_max')}",
        f"    ATR sl mult    : {sigs.get('atr_sl_mult')}",
        f"    Hard stop      : {sigs.get('hard_stop_loss_pct')}%",
        f"    Min ATR%       : {sigs.get('min_atr_pct')}",
        f"    TP mult        : {sigs.get('atr_tp_mult')}",
        "=" * 50,
    ]
    return "\n".join(lines)


def main():
    LOG.info("Diamond Agent gestart")
    last_analyze = 0.0
    last_report_day = -1

    while True:
        now = datetime.now(timezone.utc)

        # Dagrapport om 08:00 UTC
        if now.hour == REPORT_HOUR and now.day != last_report_day:
            report = build_report()
            send_email(f"Diamond Bot Dagrapport {now.strftime('%d-%m-%Y')}", report)
            last_report_day = now.day
            LOG.info("Dagrapport verstuurd")

        # Analyse elke 6 uur
        if time.time() - last_analyze >= ANALYZE_INTERVAL_SEC:
            trades  = load_trades()
            changes = analyze(trades)
            if changes:
                apply_changes(changes)
                # Stuur ook een melding bij aanpassing
                msg = "Config automatisch aangepast:\n\n"
                for k, v in changes.items():
                    msg += f"  {k}: {v}\n"
                msg += f"\n{build_report()}"
                send_email("Diamond Bot - Config aangepast", msg)
            last_analyze = time.time()

        time.sleep(60)


if __name__ == "__main__":
    main()
