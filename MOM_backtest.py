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


import math
import numpy as np
import pandas as pd


def simulate_mom5_from_df(
    df: pd.DataFrame,
    *,
    start_capital: float = 100000.0,
    lot_cost: float = 10000.0,        # cash needed to deploy 1 lot (long or short)
    qty_per_lot: int = 75,            # contracts/shares per lot
    delta: float = 0.5,               # ATM option delta proxy
    lots_when_in: int = 1,            # used when sizing='fixed'
    sizing: str = "fixed",            # 'fixed' or 'dynamic'
    use_fraction: float = 1.0,        # for dynamic sizing: fraction of deployable cash to use (0..1]
    allow_short: bool = False,        # short when roll5_pts < 0
    max_capital_to_use: float | None = None,  # cap on deployable cash per day
    max_lots: int | None = None,      # hard cap on absolute number of lots
    stop_on_bankrupt: bool = True,    # stop if equity <= 0
    mom_window: int = 5,
    name: str = "MOM5_no_lookahead"
):
    # ---- Prep & features ----
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
    df["pts"] = df["Open"] - df["prev_close"]     # pts_i = Open_i - Close_{i-1}
    df["roll_pts"] = df["pts"].rolling(mom_window).sum()  # MOM_i, known at OPEN_i

    point_value = qty_per_lot * delta

    equity = start_capital
    peak = start_capital
    equity_prev = start_capital

    rets, rows = [], []
    traded_days = 0
    max_dd = 0.0
    bankrupt = False
    bankrupt_date = None

    # Overnight position that was set yesterday (for PnL today)
    prev_overnight_pos = 0

    n = len(df)
    for i in range(n):
        date_i = idx.iloc[i]
        pts_i  = df.at[i, "pts"]                  # realized movement for *today*
        mom_i  = df.at[i, "roll_pts"]             # signal known at OPEN_i

        # 1) Realize P&L today from yesterday's overnight position (if any)
        pnl_today = 0.0
        if pd.notna(pts_i) and prev_overnight_pos != 0:
            pnl_today = pts_i * point_value * prev_overnight_pos
            traded_days += 1

        equity += pnl_today

        # Bankruptcy check after realizing P&L
        if equity <= 0 and not bankrupt:
            equity = 0.0
            bankrupt = True
            bankrupt_date = date_i

        # Update drawdown & daily return (after P&L realization)
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

        daily_ret = 0.0 if equity_prev <= 0 else (equity / equity_prev - 1.0)
        rets.append(daily_ret)
        equity_prev = equity

        # 2) Decide today's end-of-day position to carry into tomorrow
        #    (skip opening on the very last row because there's no next day to exit)
        position_to_carry = 0
        lots_traded_eod = 0

        if (i < n - 1) and (not bankrupt):
            # Desired sign from signal known this morning
            sign = 0
            if pd.notna(mom_i):
                if mom_i > 0:
                    sign = +1
                elif mom_i < 0 and allow_short:
                    sign = -1
                else:
                    sign = 0

            if sign != 0:
                # Deployable cash ceiling
                cap_ceiling = equity if max_capital_to_use is None else min(equity, max_capital_to_use)
                cap_ceiling = max(0.0, cap_ceiling)

                if sizing.lower() == "fixed":
                    abs_desired = int(max(0, lots_when_in))
                    # also must be affordable under cap_ceiling
                    abs_desired = min(abs_desired, int(cap_ceiling // lot_cost))
                elif sizing.lower() == "dynamic":
                    abs_desired = int(max(0, math.floor((use_fraction * cap_ceiling) / lot_cost)))
                else:
                    raise ValueError("sizing must be 'fixed' or 'dynamic'")

                if max_lots is not None:
                    abs_desired = min(abs_desired, int(max(0, max_lots)))

                position_to_carry = int(math.copysign(abs_desired, sign)) if abs_desired > 0 else 0

            # We’re flat intraday; at the CLOSE we open the overnight position:
            lots_traded_eod = position_to_carry

        # Store row (PnL realized today, position we will carry overnight)
        rows.append({
            "Date": date_i,
            "Pts": pts_i,
            f"MOM{mom_window}": mom_i,
            "PnL": pnl_today,                   # realized from yesterday's carry
            "Equity": equity,
            "Drawdown": dd,
            "Position": position_to_carry,      # lots carried overnight to next open
            "LotsTraded": lots_traded_eod       # executed at today's close
        })

        # The position we just set at today's close will be realized tomorrow:
        prev_overnight_pos = position_to_carry

        if bankrupt and stop_on_bankrupt:
            break

    # ---- Metrics ----
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

    if len(rows) > 0:
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
        "use_fraction": float(use_fraction),
        "lots_when_in": int(lots_when_in),
        "mom_window": int(mom_window),
    }

    timeline = pd.DataFrame(rows)
    return summary, timeline

# ----------------- Yearly restart metrics (from timeline) -----------------
def compute_yearly_restart_metrics_from_timeline(timeline: pd.DataFrame,
                                                 start_capital: float = 100000.0) -> pd.DataFrame:

    if timeline.empty:
        return pd.DataFrame()

    tl = timeline.copy()
    if not pd.api.types.is_datetime64_any_dtype(tl["Date"]):
        tl["Date"] = pd.to_datetime(tl["Date"])
    tl = tl.sort_values("Date").reset_index(drop=True)

    tl["Year"] = tl["Date"].dt.year

    out_rows = []
    for yr, g in tl.groupby("Year", sort=True):
        g = g.sort_values("Date").reset_index(drop=True).copy()

        # PnL as-if we *start fresh* on the first day of this year
        pnl_adj = g["PnL"].copy()
        if len(pnl_adj) > 0:
            pnl_adj.iloc[0] = 0.0   # no carry from prior year

        # Equity path within the year
        eq = start_capital + pnl_adj.cumsum()

        # Daily returns within the year
        eq_prev = eq.shift(1).fillna(start_capital)
        daily_ret = (eq / eq_prev - 1.0).fillna(0.0).astype(float)

        # Drawdown
        peak = eq.cummax().replace(0, np.nan)
        dd = (eq / peak - 1.0).fillna(0.0)
        max_dd = float(dd.min()) if len(dd) else 0.0

        # Risk stats
        if len(daily_ret) >= 2:
            mean_r = float(daily_ret.mean())
            std_r  = float(daily_ret.std(ddof=1)) if daily_ret.std(ddof=1) > 0 else 1e-12
            sharpe = (mean_r / std_r) * np.sqrt(252.0)
            neg = daily_ret[daily_ret < 0]
            neg_std = float(neg.std(ddof=1)) if len(neg) > 1 else (float(neg.std(ddof=0)) if len(neg) > 0 else 1e-12)
            sortino = (mean_r / (neg_std if neg_std > 0 else 1e-12)) * np.sqrt(252.0)
        else:
            sharpe = 0.0
            sortino = 0.0

        # CAGR over this (possibly partial) year
        start_d, end_d = g["Date"].iloc[0], g["Date"].iloc[-1]
        years = max(1e-9, (pd.to_datetime(end_d) - pd.to_datetime(start_d)).days / 365.25)
        end_equity = float(eq.iloc[-1])
        cagr = ((end_equity / start_capital) ** (1.0 / years) - 1.0) * 100.0 if end_equity > 0 else -100.0

        # Calmar
        calmar = (cagr / 100.0) / (abs(max_dd) + 1e-12) if max_dd < 0 else np.nan

        # Bankrupt check (inside the year)
        bankrupt_mask = eq <= 0
        bankrupt = bool(bankrupt_mask.any())
        bankrupt_date = g.loc[bankrupt_mask.idxmax(), "Date"] if bankrupt else None

        out_rows.append({
            "Year": int(yr),
            "start_date": pd.to_datetime(start_d),
            "end_date": pd.to_datetime(end_d),
            "ending_capital": end_equity,
            "total_pnl": float(pnl_adj.sum()),
            "traded_days": int((pnl_adj != 0).sum()),
            "bankrupt": bankrupt,
            "bankrupt_date": pd.to_datetime(bankrupt_date) if bankrupt else None,
            "period_days": int((pd.to_datetime(end_d) - pd.to_datetime(start_d)).days + 1),
            "CAGR_%": float(cagr),
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "Calmar": float(calmar),
            "MaxDD_%": float(max_dd * 100.0),
        })

    yr_df = pd.DataFrame(out_rows).sort_values("Year")
    return yr_df



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
            
    # ---- Per-year metrics (restart each year; non-compounded) ----
            st.subheader("Per-year metrics (restart each year, non-compounded)")
            yr_df = compute_yearly_restart_metrics_from_timeline(
                timeline,
                start_capital=float(start_capital)
            )
            if yr_df.empty:
                st.info("No yearly rows to show for the selected range.")
            else:
                # Nicely formatted view
                yr_df["Year"] = yr_df["Year"].astype(int)
                show_cols = [
                    "Year", "start_date", "end_date",
                    "ending_capital", "total_pnl", "traded_days",
                    "bankrupt", "bankrupt_date", "period_days",
                    "CAGR_%", "Sharpe", "Sortino", "Calmar", "MaxDD_%"
                ]
                yr_view = yr_df[show_cols].copy()
                st.dataframe(
                    yr_view,
                    use_container_width=True,
                    column_config={
                        "start_date": st.column_config.DatetimeColumn(format="DD-MM-YYYY"),
                        "end_date": st.column_config.DatetimeColumn(format="DD-MM-YYYY"),
                        "bankrupt_date": st.column_config.DatetimeColumn(format="DD-MM-YYYY"),
                    }
                )
                st.download_button(
                    "Download per-year metrics CSV",
                    yr_view.to_csv(index=False).encode(),
                    file_name=f"{symbol}_yearly_restart_metrics.csv",
                    mime="text/csv"
                )

            st.subheader("Equity curve")
            timeline = timeline.sort_values("Date")
            st.line_chart(timeline.set_index("Date")["Equity"])
            st.subheader("Daily Data (Open/Close + Pts & MOM)")
            st.dataframe(
                show.round(3).tail(250),
                use_container_width=True,
                column_config={"Date": st.column_config.DatetimeColumn(format="DD-MM-YYYY")}
            )
            csv_daily = show.to_csv(index=False).encode()
            st.download_button("Download daily table CSV", csv_daily, file_name=f"{symbol}_daily_mom.csv", mime="text/csv")


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
