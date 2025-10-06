# mom5_dashboard.py
# Streamlit app: MOM5 downloader + backtester (date-range, daily table shown)

import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date, timedelta

# ---- MUST be the first Streamlit call ----
st.set_page_config(page_title="MOM5 Backtest", page_icon="📈", layout="wide")

# ----------------- MOM5 simulator -----------------
def simulate_mom5_from_df(
    df: pd.DataFrame,
    *,
    start_capital: float = 100000.0,
    lot_cost: float = 10000.0,
    qty_per_lot: int = 75,
    delta: float = 0.5,
    lots_when_in: int = 1,
    sizing: str = "fixed",           # 'fixed' | 'dynamic'
    use_fraction: float = 1.0,       # for dynamic sizing
    allow_short: bool = False,
    max_capital_to_use: float | None = None,
    max_lots: int | None = None,
    stop_on_bankrupt: bool = True,
    name: str = "MOM5",
    mom_window: int = 5,
):
    df = df.copy()
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        idx = df["Date"]
    else:
        df = df.sort_index()
        idx = df.index

    df["prev_close"] = df["Close"].shift(1)
    df["pts"] = df["Open"] - df["prev_close"]
    df["roll_pts"] = df["pts"].rolling(mom_window).sum()

    point_value = qty_per_lot * delta

    equity = start_capital
    peak = start_capital
    equity_prev = start_capital

    rets, rows = [], []
    traded_days = 0
    max_dd = 0.0
    bankrupt = False
    bankrupt_date = None
    prev_position = 0

    for i, r in df.iterrows():
        d = pd.to_datetime(idx[i])
        pts = r.get("pts", np.nan)
        s   = r.get("roll_pts", np.nan)

        sign = 0
        if pd.notna(s) and pd.notna(pts):
            if s > 0:
                sign = +1
            elif s < 0 and allow_short:
                sign = -1

        abs_desired = 0
        if sign != 0:
            if sizing.lower() == "fixed":
                abs_desired = int(max(0, lots_when_in))
            elif sizing.lower() == "dynamic":
                cap_cap = equity if max_capital_to_use is None else min(equity, max_capital_to_use)
                cap_cap = max(0.0, cap_cap)
                abs_desired = int(max(0, math.floor((use_fraction * cap_cap) / lot_cost)))
            else:
                raise ValueError("sizing must be 'fixed' or 'dynamic'")
            if max_lots is not None:
                abs_desired = min(abs_desired, int(max(0, max_lots)))

        position = int(math.copysign(abs_desired, sign)) if abs_desired > 0 else 0
        lots_delta = position - prev_position

        pnl = 0.0
        if pd.notna(pts) and position != 0:
            pnl = pts * point_value * position
            traded_days += 1
        equity += pnl

        if equity <= 0 and not bankrupt:
            bankrupt = True
            equity = 0.0
            bankrupt_date = d

        peak = max(peak, equity)
        dd = (equity / peak - 1.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

        daily_ret = 0.0 if equity_prev <= 0 else (equity / equity_prev - 1.0)
        rets.append(daily_ret)
        equity_prev = equity

        rows.append({
            "Date": d,
            "Pts": pts,
            f"MOM{mom_window}": s,
            "Position": position,
            "LotsTraded": lots_delta,
            "PnL": pnl,
            "Equity": equity,
            "Drawdown": dd
        })

        prev_position = 0
        if bankrupt and stop_on_bankrupt:
            break

    rets = np.array(rets, dtype=float)
    if len(rets) < 2:
        sharpe = sortino = 0.0
    else:
        mean_r = rets.mean()
        std_r  = rets.std(ddof=1) or 1e-12
        sharpe = (mean_r / std_r) * np.sqrt(252)
        neg = rets[rets < 0]
        neg_std = (neg.std(ddof=1) if len(neg) > 1 else (neg.std(ddof=0) if len(neg) > 0 else 0.0)) or 1e-12
        sortino = (mean_r / neg_std) * np.sqrt(252)

    max_dd_pct = max_dd * 100.0
    total_pnl = equity - start_capital

    if rows:
        start_date = rows[0]["Date"]
        end_date   = rows[-1]["Date"]
        years = max(1e-9, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25)
    else:
        years = 1e-9

    cagr = ((equity / start_capital) ** (1 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    calmar = (cagr / 100.0) / (abs(max_dd) + 1e-12) if max_dd < 0 else np.nan

    summary = {
        "strategy": name,
        "ending_capital": float(equity),
        "total_pnl": float(total_pnl),
        "traded_days": int(traded_days),
        "bankrupt": bool(bankrupt),
        "bankrupt_date": pd.to_datetime(bankrupt_date) if bankrupt_date is not None else None,
        "CAGR_%": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "MaxDD_%": float(max_dd_pct),
        "sizing": sizing,
    }
    timeline = pd.DataFrame(rows)
    return summary, timeline

# ----------------- UI -----------------
st.title("📈 MOM5 Backtest — NIFTY/BANKNIFTY & NIFTY50")

NIFTY50 = [
    "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","ITC","LT","SBIN","BHARTIARTL","HINDUNILVR",
    "KOTAKBANK","BAJFINANCE","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA","HCLTECH","TITAN","ULTRACEMCO","TATASTEEL",
    "WIPRO","ONGC","NTPC","TATAMOTORS","POWERGRID","ADANIENT","ADANIPORTS","COALINDIA","M&M","BAJAJ-AUTO",
    "TECHM","HDFCLIFE","NESTLEIND","GRASIM","BRITANNIA","CIPLA","DRREDDY","JSWSTEEL","HEROMOTOCO","EICHERMOT",
    "DIVISLAB","HINDALCO","SBILIFE","BPCL","UPL","INDUSINDBK","TATACONSUM","APOLLOHOSP","LTIM","BAJAJFINSV"
]
DISPLAY_TO_SYMBOL = {
    "NIFTY 50 Index": "^NSEI",
    "BANKNIFTY Index": "^NSEBANK",
    **{f"{s}.NS": f"{s}.NS" for s in NIFTY50},
    "Custom…": "CUSTOM",
}

# --- top controls ---
c = st.columns(4)
with c[0]:
    choice = st.selectbox("Symbol", list(DISPLAY_TO_SYMBOL.keys()), index=0)
symbol = DISPLAY_TO_SYMBOL[choice]
if symbol == "CUSTOM":
    symbol = st.text_input("Enter Yahoo symbol", value="HDFCBANK.NS")

with c[1]:
    mom_window = st.number_input("MOM window", 3, 250, 5, 1)

# Date range (dd-mm-yyyy to dd-mm-yyyy) — no limit
today = date.today()
default_start = today - timedelta(days=365)
d1, d2 = st.columns(2)
with d1:
    start_date = st.date_input(
        "Start date (dd-mm-yyyy)",
        default_start,
        min_value=date(1990, 1, 1),     # 👈 allows dates from 1990 onwards
        max_value=today,
        format="DD-MM-YYYY"
    )
with d2:
    end_date = st.date_input(
        "End date (dd-mm-yyyy)",
        today,
        min_value=date(1990, 1, 1),
        max_value=today,
        format="DD-MM-YYYY"
    )

with st.expander("Backtest params", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sizing = st.selectbox("Sizing", ["fixed", "dynamic"], index=0)
        lots_when_in = st.number_input("Lots when in (fixed)", 1, 100000, 1, 1)
        use_fraction = st.slider("Use fraction (dynamic)", 0.0, 1.0, 1.0, 0.05)
    with c2:
        start_capital = st.number_input("Start capital", 0.0, 1e12, 100000.0, 1000.0)
        max_capital_to_use = st.number_input("Max capital to deploy", 0.0, 1e12, 1_000_000.0, 10000.0)
        max_lots = st.number_input("Max lots", 0, 100000, 100, 1)
    with c3:
        lot_cost = st.number_input("Lot cost", 0.0, 1e12, 10000.0, 100.0)
        qty_per_lot = st.number_input("Qty per lot", 1, 1_000_000, 75, 1)
        delta = st.number_input("Delta", 0.0, 1.0, 0.5, 0.05)
    with c4:
        allow_short = st.checkbox("Allow short", value=False)
        stop_on_bankrupt = st.checkbox("Stop on bankrupt", value=True)

run = st.button("Run backtest")

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

if run:
    try:
        raw = yf.download(
            symbol,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # make end inclusive
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True
        )
        if raw.empty:
            st.error("No data returned. Check symbol or dates.")
        else:
            prices = flatten_cols(raw)[["Open", "Close"]].dropna().copy()
            prices = prices.reset_index().rename(columns={"Date": "Date"})
            # Compute daily Pts and MOM for display table
            show = prices.copy()
            show["prev_close"] = show["Close"].shift(1)
            show["Pts"] = show["Open"] - show["prev_close"]
            show[f"MOM{mom_window}"] = show["Pts"].rolling(int(mom_window)).sum()
            show = show.drop(columns=["prev_close"])
            st.subheader("Daily Data (Open/Close + Pts & MOM)")
            st.dataframe(
                show.round(3).tail(250),
                use_container_width=True,
                column_config={"Date": st.column_config.DatetimeColumn(format="DD-MM-YYYY")}
            )
            csv_daily = show.to_csv(index=False).encode()
            st.download_button("Download daily table CSV", csv_daily, file_name=f"{symbol}_daily_mom.csv", mime="text/csv")

            # Backtest
            summary, timeline = simulate_mom5_from_df(
                df=prices,
                start_capital=float(start_capital),
                lot_cost=float(lot_cost),
                qty_per_lot=int(qty_per_lot),
                delta=float(delta),
                lots_when_in=int(lots_when_in),
                sizing=sizing,
                use_fraction=float(use_fraction),
                allow_short=allow_short,
                max_capital_to_use=float(max_capital_to_use),
                max_lots=int(max_lots),
                stop_on_bankrupt=stop_on_bankrupt,
                name=f"MOM{mom_window}",
                mom_window=int(mom_window),
            )

            st.subheader("Summary")
            st.write(pd.DataFrame([summary]))

            st.subheader("Equity curve")
            timeline = timeline.sort_values("Date")
            st.line_chart(timeline.set_index("Date")["Equity"])

            st.subheader("Backtest rows (tail)")
            st.dataframe(
                timeline.tail(200),
                use_container_width=True,
                column_config={"Date": st.column_config.DatetimeColumn(format="DD-MM-YYYY")}
            )
            csv_tl = timeline.to_csv(index=False).encode()
            st.download_button("Download backtest timeline CSV", csv_tl, file_name=f"{symbol}_timeline.csv", mime="text/csv")

    except Exception as e:
        st.exception(e)
