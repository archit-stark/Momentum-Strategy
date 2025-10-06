# mom5_app.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="MOM5 Signal", page_icon="📈", layout="centered")
st.title("📈 MOM5 Option Bias (Open vs Prev Close)")
st.caption("Signal: sum of last 5 days' (Open - prev Close). >0 = LONG, <0 = SHORT, 0 = FLAT")

# --------- Symbols (NIFTY, BANKNIFTY + NIFTY50) ----------
# Yahoo Finance symbols; composition can change over time.
SYMBOLS = {
    "📊 NIFTY 50 Index": "^NSEI",
    "🏦 NIFTY Bank": "^NSEBANK",
    # NIFTY50 (common symbols)
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Grasim": "GRASIM.NS",
    "Hindalco": "HINDALCO.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Britannia": "BRITANNIA.NS",
    "Nestle India": "NESTLEIND.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "Titan": "TITAN.NS",
    "BEL": "BEL.NS",
    "LTIMindtree": "LTIM.NS",
    "ONGC": "ONGC.NS",  # duplicate safe-guard
}

# A friendly list for the selectbox
OPTIONS = ["📊 NIFTY 50 Index", "🏦 NIFTY Bank"] + \
          sorted([k for k in SYMBOLS.keys() if k not in {"📊 NIFTY 50 Index", "🏦 NIFTY Bank"}]) + \
          ["Custom…"]

# --------- UI controls ----------
col1, col2, col3 = st.columns([1.4, 1, 1])
with col1:
    choice = st.selectbox("Choose symbol", OPTIONS, index=0,
                          help="Pick from NIFTY/BANKNIFTY or any NIFTY50 stock. Select 'Custom…' to type a Yahoo symbol.")
    if choice == "Custom…":
        ticker = st.text_input("Enter Yahoo symbol", value="^NSEI",
                               help="e.g. ^NSEI, HDFCBANK.NS, TCS.NS")
    else:
        ticker = SYMBOLS[choice]

with col2:
    lookback_days = st.number_input("Download days", min_value=10, max_value=365, value=30, step=1)
with col3:
    mom_window = st.number_input("MOM window", min_value=3, max_value=20, value=5, step=1)

run = st.button("Get MOM5 Signal")

# --------- Helpers ----------
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def compute_mom5(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    df = flatten_cols(df)
    df = df[['Open', 'Close']].copy()
    df['prev_close'] = df['Close'].shift(1)
    df['pts'] = df['Open'] - df['prev_close']
    df['mom'] = df['pts'].rolling(window).sum()
    return df

def signal_text(value: float) -> str:
    if pd.isna(value):
        return "Insufficient data"
    if value > 0:
        return "LONG (buy call / long bias)"
    if value < 0:
        return "SHORT (buy put / short bias)"
    return "FLAT / NO TRADE"

# --------- Action ----------
if run:
    try:
        raw = yf.download(
            ticker, period=f"{int(lookback_days)}d", interval="1d",
            auto_adjust=False, progress=False, threads=True
        )
        if raw.empty or len(raw) < mom_window + 1:
            st.error("Not enough data. Try a larger download window or check the symbol.")
        else:
            df = compute_mom5(raw, window=mom_window).dropna().copy()
            latest_row = df.iloc[-1]
            mom_val = float(latest_row['mom'])
            sig = signal_text(mom_val)

            color = "green" if "LONG" in sig else ("red" if "SHORT" in sig else "gray")
            st.markdown(
                f"""
                <div style='text-align:center;'>
                    <h2>{choice if choice!='Custom…' else ticker} — MOM{mom_window}</h2>
                    <h3 style='color:{color}; font-weight:700;'>{sig}</h3>
                    <p style='font-size:20px; margin-top:-5px;'>MOM value: {mom_val:,.2f} points</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Recent pts (Open − prev Close)")
            show = df[['pts', 'mom']].tail(10).round(2).rename(
                columns={'pts': 'Pts', 'mom': f'MOM{mom_window}'}
            )
            show.index.name = "Date"
            st.dataframe(show, use_container_width=True)

            st.caption(
                "Note: For ATM options, rough per-lot P&L ≈ pts × (Δ × lot_qty). "
                "For Δ≈0.5 and lot size=75, per-point per-lot ≈ 37.5."
            )
    except Exception as e:
        st.exception(e)
