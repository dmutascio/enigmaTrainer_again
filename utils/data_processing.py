import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from ta import trend, momentum, volatility, volume, add_all_ta_features
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import os
import dotenv
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="ta")

dotenv.load_dotenv()

# Setup Alpaca client
API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

def engineer_features(data):
    """
    Calculate technical indicators and other features.
    """

    # Add all technical indicators
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume",
        fillna=True
    )

    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['returns_5'] = data['close'].pct_change(5)
    data['returns_15'] = data['close'].pct_change(15)

    return data

# Function to get and preprocess data
def get_data(symbol, start_date, end_date):
    """
    Fetches historical stock data and calculates various technical indicators for a given symbol.

    This function retrieves minute-by-minute stock data from Alpaca for the specified symbol
    and date range. It then calculates a set of technical indicators and features, including
    moving averages, RSI, Bollinger Bands, MACD, and more. It also fetches market data (using SPY as a proxy)
    to calculate market returns.
    """

    # Fetch data
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute, # type: ignore
        start=start_date,
        end=end_date
    )

    # Show loader
    with tqdm(total=1, desc=f'[{datetime.now().strftime("%H:%M:%S")}] Fetching market data {symbol}') as pbar:
        bars = stock_client.get_stock_bars(request_params)
        pbar.update(1)

    # Convert to DataFrame
    data = bars.df.reset_index() # type: ignore
    data = data.set_index('timestamp')

    # Calculate features
    data = engineer_features(data)

    # Calculate target (5-minute forward return)
    data['target'] = data['close'].pct_change(5).shift(-5)


    # Fetch market data using SPY as sp500 proxy
    # Show loader
    # FIXME: use DXY for usd proxy?
    market_request_params = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute, # type: ignore
        start=start_date,
        end=end_date
    )
    with tqdm(total=1, desc=f'[{datetime.now().strftime("%H:%M:%S")}] Fetching market data SPY') as pbar:
        market_bars = stock_client.get_stock_bars(market_request_params)
        pbar.update(1)
    market_data = market_bars.df.reset_index().set_index('timestamp') # type: ignore
    data['market_returns'] = market_data['close'].pct_change().reindex(data.index).fillna(0)

    # # Drop NaN values
    data = data.dropna()

    if 'symbol' in data.columns:
        data = data.drop('symbol', axis=1)

    # Separate features and target
    features = [col for col in data.columns if col not in ['target']]

    X = data[features].values
    y = data['target'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, data

def fetch_market_data(start_date, end_date):
    market_request_params = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute, # type: ignore
        start=start_date,
        end=end_date
    )
    with tqdm(total=1, desc='Fetching market data SPY') as pbar:
        market_bars = stock_client.get_stock_bars(market_request_params)
        pbar.update(1)
    return market_bars.df.reset_index().set_index('timestamp') # type: ignore
