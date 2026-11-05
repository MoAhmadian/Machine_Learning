import json
import os
import smtplib
from datetime import date, datetime, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path

import yaml
import yfinance as yf

CONFIG = Path("watchlist.yml")
STATE = Path(".state/alerts.json")

GREEN, RED, AMBER, GRAY = "#16a34a", "#dc2626", "#d97706", "#6b7280"


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(state):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state))


def get_prices(symbol):
    """Return (current_price, previous_close)."""
    hist = yf.Ticker(symbol).history(period="5d", interval="1d")["Close"].dropna()
    if len(hist) < 2:
        raise ValueError("not enough price history")
    return float(hist.iloc[-1]), float(hist.iloc[-2])


def money(x):
    return f"${x:,.2f}"


def color_for(change):
    return GREEN if change >= 0 else RED


def build_email(alerts, snapshot):
    """Return (subject, plain_text, html). Alerts/snapshot are lists of dicts."""
    symbols = list(dict.fromkeys(a["symbol"] for a in alerts))
    subject = f"Stock alert: {', '.join(symbols)}"
    now = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")

    # ---- plain-text fallback ----
    text = [f"stock_checker alerts ({now})", ""]
    for a in alerts:
        text.append(f"- {a['label']}: {a['symbol']} {money(a['price'])} "
                    f"({a['change']:+.2f}% vs prev close {money(a['prev'])}); {a['detail']}")
    text += ["", "Watchlist snapshot:"]
    for s in snapshot:
        text.append(f"  {s['symbol']:<8} {money(s['price']):>12}  {s['change']:+.2f}%")
    plain = "\n".join(text)

    # ---- HTML ----
    th = "padding:8px 10px;text-align:left;font-size:12px;color:#6b7280;border-bottom:2px solid #e5e7eb;"
    td = "padding:10px;border-bottom:1px solid #f1f5f9;font-size:14px;"

    alert_rows = ""
    for a in alerts:
        badge_color = RED if a["kind"] == "below_min" else (GREEN if a["change"] >= 0 else RED)
        arrow = "▲" if a["change"] >= 0 else "▼"
        alert_rows += f"""
        <tr>
          <td style="{td}"><a href="https://finance.yahoo.com/quote/{escape(a['symbol'])}"
              style="color:#111827;font-weight:700;text-decoration:none;">{escape(a['symbol'])}</a></td>
          <td style="{td}"><span style="background:{badge_color};color:#fff;border-radius:10px;
              padding:2px 8px;font-size:12px;white-space:nowrap;">{escape(a['label'])}</span></td>
          <td style="{td}text-align:right;font-weight:600;">{money(a['price'])}</td>
          <td style="{td}text-align:right;color:{color_for(a['change'])};font-weight:600;">
              {arrow} {a['change']:+.2f}%</td>
          <td style="{td}text-align:right;color:{GRAY};">{escape(a['detail'])}</td>
        </tr>"""

    snap_rows = ""
    for s in snapshot:
        snap_rows += f"""
        <tr>
          <td style="{td}font-weight:600;">{escape(s['symbol'])}</td>
          <td style="{td}text-align:right;">{money(s['price'])}</td>
          <td style="{td}text-align:right;color:{color_for(s['change'])};">{s['change']:+.2f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f3f4f6;
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:24px 0;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0"
             style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;">
        <tr><td style="background:#111827;padding:20px 24px;">
          <div style="color:#ffffff;font-size:20px;font-weight:700;">📈 Stock alert</div>
          <div style="color:#9ca3af;font-size:13px;margin-top:4px;">
            {len(alerts)} trigger{'s' if len(alerts) != 1 else ''} · {now}</div>
        </td></tr>
        <tr><td style="padding:20px 24px 8px;">
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
            <tr><th style="{th}">Ticker</th><th style="{th}">Alert</th>
                <th style="{th}text-align:right;">Price</th>
                <th style="{th}text-align:right;">vs prev close</th>
                <th style="{th}text-align:right;">Detail</th></tr>
            {alert_rows}
          </table>
        </td></tr>
        <tr><td style="padding:16px 24px 8px;">
          <div style="font-size:13px;font-weight:700;color:#374151;margin-bottom:6px;">Watchlist snapshot</div>
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
            <tr><th style="{th}">Ticker</th><th style="{th}text-align:right;">Price</th>
                <th style="{th}text-align:right;">Change</th></tr>
            {snap_rows}
          </table>
        </td></tr>
        <tr><td style="padding:16px 24px 24px;color:#9ca3af;font-size:12px;">
          Prices from Yahoo Finance and may be delayed ~15 min. Change is vs previous day's close.
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>"""
    return subject, plain, html


def send_email(subject, plain, html):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_TO"]
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def main():
    stocks = yaml.safe_load(CONFIG.read_text())["stocks"]
    today = date.today().isoformat()
    # keep only today's dedupe keys so the state file stays tiny
    state = {k: v for k, v in load_state().items() if k.endswith(today)}
    alerts, snapshot = [], []

    for s in stocks:
        symbol = s["symbol"]
        try:
            price, prev_close = get_prices(symbol)
        except Exception as e:
            print(f"{symbol}: skipped ({e})")
            continue

        change = (price - prev_close) / prev_close * 100
        print(f"{symbol}: {price:.2f} ({change:+.2f}% vs prev close {prev_close:.2f})")
        snapshot.append({"symbol": symbol, "price": price, "change": change})

        checks = []
        if "min_price" in s and price < s["min_price"]:
            checks.append(("below_min", "Below minimum", f"min {money(s['min_price'])}"))
        if "change_pct" in s and abs(change) >= s["change_pct"]:
            checks.append(("big_move", "Big move", f"limit ±{s['change_pct']}%"))

        for kind, label, detail in checks:
            key = f"{symbol}:{kind}:{today}"
            if key not in state:  # alert once per ticker/type/day
                alerts.append({"symbol": symbol, "kind": kind, "label": label,
                               "price": price, "prev": prev_close,
                               "change": change, "detail": detail})
                state[key] = True

    if alerts:
        send_email(*build_email(alerts, snapshot))
        print(f"Sent email with {len(alerts)} alert(s)")
    else:
        print("No new alerts")

    save_state(state)


if __name__ == "__main__":
    main()
