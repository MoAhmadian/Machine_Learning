import json
import os
import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import yaml
import yfinance as yf

CONFIG = Path("watchlist.yml")
STATE = Path(".state/alerts.json")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(state):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state))


def get_prices(symbol):
    """Return (current_price, previous_close, last_bar_date)."""
    hist = yf.Ticker(symbol).history(period="5d", interval="1d")["Close"].dropna()
    if len(hist) < 2:
        raise ValueError("not enough price history")
    return float(hist.iloc[-1]), float(hist.iloc[-2]), hist.index[-1].date().isoformat()


def send_email(subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_TO"]
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def main():
    stocks = yaml.safe_load(CONFIG.read_text())["stocks"]
    today = date.today().isoformat()
    # keep only today's dedupe keys so the state file stays tiny
    state = {k: v for k, v in load_state().items() if k.endswith(today)}
    alerts = []

    for s in stocks:
        symbol = s["symbol"]
        try:
            price, prev_close, bar_date = get_prices(symbol)
        except Exception as e:
            print(f"{symbol}: skipped ({e})")
            continue

        change = (price - prev_close) / prev_close * 100
        print(f"{symbol}: {price:.2f} ({change:+.2f}% vs prev close {prev_close:.2f})")

        checks = []
        if "min_price" in s and price < s["min_price"]:
            checks.append(("below_min", f"{symbol} fell below minimum: "
                           f"${price:.2f} < ${s['min_price']:.2f}"))
        if "change_pct" in s and abs(change) >= s["change_pct"]:
            checks.append(("big_move", f"{symbol} moved {change:+.2f}% vs previous close "
                           f"(${prev_close:.2f} -> ${price:.2f}), threshold {s['change_pct']}%"))

        for kind, text in checks:
            key = f"{symbol}:{kind}:{today}"
            if key not in state:  # alert once per ticker/type/day
                alerts.append(text)
                state[key] = True

    if alerts:
        send_email(f"Stock alert: {len(alerts)} trigger(s)", "\n".join(alerts))
        print(f"Sent email with {len(alerts)} alert(s)")
    else:
        print("No new alerts")

    save_state(state)


if __name__ == "__main__":
    main()
