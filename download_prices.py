import pandas as pd
import yfinance as yf

# Pull the equity history once and park it in a CSV
# Run this from the notebook folder, then rely on the cached file
tickers = ["VOW3.DE", "DBK.DE", "TTE.PA", "SIE.DE", "TEF.MC"]
out_file = "equity_prices_daily.csv"

# Daily closes stored rather than weekly
px = yf.download(tickers, period="3y", interval="1d", progress=False, auto_adjust=True)
px = px["Close"][tickers]

assert not px.empty, "download returned nothing - check the tickers and your connection"

# Exact original terminal output formatting preserved
print(f"{px.shape[0]} rows x {px.shape[1]} tickers")
print(f"{px.index.min():%Y-%m-%d} to {px.index.max():%Y-%m-%d}\n")
print("missing closes per ticker (holiday calendars differ by exchange):")
print(px.isna().sum().to_string())

px.to_csv(out_file)
print(f"\nsaved -> {out_file}")
