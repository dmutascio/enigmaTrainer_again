import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from ta import trend, momentum, volatility, volume
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from torch.multiprocessing import Pool, cpu_count
from functools import partial
import dotenv
import os
import multiprocessing
from tqdm import tqdm

import sys


# Alpaca API setup
dotenv.load_dotenv()
API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']

# Initialize Alpaca clients
stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        x = self.fc(x[:, -1, :])
        return x

# Custom dataset
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_length],
                self.data[index+self.seq_length, 0])

# Function to get and preprocess data
def get_data(symbol, start_date, end_date):

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
    with tqdm(total=1, desc=f'[{datetime.now().strftime("%H:%M:%S")}] Fetching market data {symbol}') as pbar:
        market_bars = stock_client.get_stock_bars(market_request_params)
        pbar.update(1)
    market_data = market_bars.df.reset_index().set_index('timestamp') # type: ignore
    data['market_returns'] = market_data['close'].pct_change().reindex(data.index).fillna(0)

    # Drop NaN values
    data = data.dropna()

    # Select features
    X = data[features].values
    y = data['returns'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, data

# Training function
def train_model(X, y, seq_length, model_params, lr, num_epochs, model_type='lstm'):
    dataset = StockDataset(np.column_stack((y.reshape(-1, 1), X)), seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    if model_type == 'lstm':
        model = LSTMModel(X.shape[1] + 1, **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(X.shape[1] + 1, **model_params)
    else:
        raise ValueError("Invalid model type")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    pbar = tqdm(range(num_epochs), desc=f'[{datetime.now().strftime("%H:%M:%S")}] Training {model_type.upper()} model')
    for epoch in pbar:
        model.train()
        total_loss = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

    return model

# Trading function
def trade(model, data, scaler, symbol, initial_balance=100000):
    model.eval()
    balance = initial_balance
    position = 0
    returns = []


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pbar = tqdm(range(len(data) - 60), desc=f'[{datetime.now().strftime("%H:%M:%S")}] Trading {symbol}')

    with torch.no_grad():
        for i in pbar:
        #for i in range(len(data) - 60):
            seq = torch.FloatTensor(data[i:i+60]).unsqueeze(0).to(device)
            prediction = model(seq).item()

            if prediction > 0.001 and position == 0:
                # Buy
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=balance // data[i+60, 0],
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(market_order_data)
                position = balance // data[i+60, 0]
                balance = 0
            elif prediction < -0.001 and position > 0:
                # Sell
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=position,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(market_order_data)
                balance = position * data[i+60, 0]
                position = 0

            # Calculate returns
            total_value = balance + position * data[i+60, 0]
            returns.append((total_value / initial_balance) - 1)

            pbar.set_postfix({'Returns': f'{returns[-1]:.2%}'})

    return returns

# Save model to disc
def save_model(model, model_type, symbol):
    print(f"Saving model: {symbol}_{model_type}_model.pth")

    torch.save(model.state_dict(), f"{symbol}_{model_type}_model.pth")

# Function to load model
def load_model(model_type, symbol, model_params):
    if model_type == 'lstm':
        model = LSTMModel(model_params['input_dim'], **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(model_params['input_dim'], **model_params)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(f"{symbol}_{model_type}_model.pth"))
    return model

# Ensemble prediction
def ensemble_predict(lstm_model, transformer_model, data):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seq = torch.FloatTensor(data).unsqueeze(0).to(device)

    lstm_pred = lstm_model(seq).item()
    transformer_pred = transformer_model(seq).item()

    return (lstm_pred + transformer_pred) / 2

# Parallel processing for multiple stocks
def process_stock(symbol, start_date, end_date, seq_length, lstm_params, transformer_params, lr, num_epochs):
    X, y, scaler, data = get_data(symbol, start_date, end_date)

    lstm_model = train_model(X, y, seq_length, lstm_params, lr, num_epochs, 'lstm')
    transformer_model = train_model(X, y, seq_length, transformer_params, lr, num_epochs, 'transformer')

    save_model(lstm_model, 'lstm', symbol)
    save_model(transformer_model, 'transformer', symbol)

    returns = trade(lstm_model, np.column_stack((y.reshape(-1, 1), X)), scaler, symbol) # type: ignore

    total_return = returns[-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

    return symbol, total_return, sharpe_ratio

# Main execution
if __name__ == "__main__":
    #symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # Stocks
    symbols = ['AAPL']  # Stocks TESTING

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use one year of data

    seq_length = 60
    lstm_params = {'hidden_dim': 128, 'num_layers': 3, 'output_dim': 1}
    transformer_params = {'d_model': 128, 'nhead': 8, 'num_layers': 3, 'output_dim': 1}
    lr = 0.001
    num_epochs = 100

    # Parallel processing
    with Pool(cpu_count()) as p:
        results = p.map(partial(process_stock, start_date=start_date, end_date=end_date,
                                seq_length=seq_length, lstm_params=lstm_params,
                                transformer_params=transformer_params, lr=lr, num_epochs=num_epochs), symbols)

    for symbol, total_return, sharpe_ratio in results:
        print(f"{symbol}:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print()
