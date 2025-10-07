@echo off
REM === Activate Anaconda base environment and run Streamlit ===
cd /d "C:\Users\archi\OneDrive\Desktop\Python"

call "C:\Users\archi\anaconda3\Scripts\activate.bat" base

streamlit run "C:\Users\archi\OneDrive\Desktop\Python\MOM_backtest.py" --server.headless=false
