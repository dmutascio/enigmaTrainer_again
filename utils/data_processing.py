import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from ta import trend, momentum, volatility, volume
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import os
import dotenv

dotenv.load_dotenv()

# Setup Alpaca client
API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Function to get and preprocess data
def get_data(symbol, start_date, end_date):
    """
    Fetches historical stock data and calculates various technical indicators for a given symbol.

    This function retrieves minute-by-minute stock data from Alpaca for the specified symbol
    and date range. It then calculates a set of technical indicators and features, including
    moving averages, RSI, Bollinger Bands, MACD, and more. It also fetches market data (using SPY as a proxy)
    to calculate market returns.

    Args:
        symbol (str): The stock symbol to fetch data for.
        start_date (datetime): The start date for the data range.
        end_date (datetime): The end date for the data range.

    Returns:
        tuple: A tuple containing four elements:
            - X (numpy.ndarray): A 2D array of normalized feature values.
            - y (numpy.ndarray): A 1D array of stock returns.
            - scaler (sklearn.preprocessing.StandardScaler): The scaler used to normalize the features.
            - data (pandas.DataFrame): The raw data including all calculated features.

    Features calculated:
        - returns: Percentage change in closing price.
        - log_returns: Natural log of returns.
        - ma_10, ma_30: 10 and 30-day moving averages.
        - rsi: Relative Strength Index.
        - bb_high, bb_mid, bb_low: Bollinger Bands.
        - macd: Moving Average Convergence Divergence.
        - atr: Average True Range.
        - volume: Trading volume.
        - obv: On-Balance Volume.
        - cci: Commodity Channel Index.
        - adx: Average Directional Index.
        - sentiment: Random values (placeholder for sentiment analysis).
        - market_returns: Returns of the overall market (using SPY as proxy).

    Note:
        - This function requires an active Alpaca API key and secret.
        - It uses the tqdm library to display progress bars during data fetching and processing.
        - NaN values are dropped from the final dataset.

    Raises:
        Any exceptions raised by the Alpaca API or data processing libraries.

    Example:
        >>> start = datetime(2023, 1, 1)
        >>> end = datetime(2023, 12, 31)
        >>> X, y, scaler, data = get_data('AAPL', start, end)
        >>> print(f"Fetched {len(data)} data points with {X.shape[1]} features.")
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

    # Create features and show loader
    features = ['returns', 'log_returns', 'ma_10', 'ma_30', 'rsi', 'bb_high', 'bb_mid', 'bb_low', 'macd', 'atr', 'volume', 'obv', 'cci', 'adx', 'sentiment', 'market_returns']
    for feature in tqdm(features, desc=f'[{datetime.now().strftime("%H:%M:%S")}] Calculating features for {symbol}'):
        match feature:
            case 'returns':
                data['returns'] = data['close'].pct_change()
            case 'log_returns':
                data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            case 'ma_10':
                data['ma_10'] = trend.sma_indicator(data['close'], window=10)
            case 'ma_30':
                data['ma_30'] = trend.sma_indicator(data['close'], window=30)
            case 'rsi':
                data['rsi'] = momentum.rsi(data['close'], window=14)
            case 'bb_high':
                data['bb_high'] = volatility.bollinger_hband(data['close'])
            case 'bb_mid':
                data['bb_mid'] = volatility.bollinger_mavg(data['close'])
            case 'bb_low':
                data['bb_low'] = volatility.bollinger_lband(data['close'])
            case 'macd':
                data['macd'] = trend.macd_diff(data['close'])
            case 'atr':
                data['atr'] = volatility.average_true_range(data['high'], data['low'], data['close'])
            case 'volume':
                pass  # Volume is already in the data
            case 'obv':
                data['obv'] = volume.on_balance_volume(data['close'], data['volume'])
            case 'cci':
                data['cci'] = trend.cci(data['high'], data['low'], data['close'])
            case 'adx':
                data['adx'] = trend.adx(data['high'], data['low'], data['close'])
            case 'sentiment':
                data['sentiment'] = np.random.randn(len(data))

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

    # Log the first few rows of the selected features
    logging.info("Selected features (first 5 rows):")
    logging.info(data[features].head())

    # Log the list of feature names
    logging.info("List of features:")
    logging.info(features)

    X = data[features].values
    y = data['returns'].values

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
