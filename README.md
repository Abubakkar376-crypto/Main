# Main 
import yfinance as yf

# Download stock data for Google (GOOG) from Yahoo Finance
stock_data = yf.download('GOOG', start='2010-01-01', end='2023-09-01')
print(stock_data.head())
