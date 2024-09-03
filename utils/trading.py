"""
This module handles the trading simulation and model inference.

It includes functions for loading trained models, performing inference,
and simulating trading based on model predictions.
"""

import torch
import multiprocessing
from functools import partial
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from datetime import datetime, timedelta
from data_processing import get_data
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import LSTM_PARAMS, TRANSFORMER_PARAMS, TRAINING_PARAMS, SYMBOLS
import logging
import dotenv
import os

dotenv.load_dotenv()

# Initialize Alpaca clients
API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Function to load model
def load_model(model_type, symbol, model_params):
    """
    Load a trained model from a file.

    Args:
        model_type (str): Type of the model ('lstm' or 'transformer').
        symbol (str): Stock symbol.
        model_params (dict): Parameters for the model.

    Returns:
        torch.nn.Module: Loaded model.

    Raises:
        ValueError: If an invalid model type is provided.
    """

    file_path = f"../trained_models/{symbol}/{symbol}_{model_type}_model.pth"

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(file_path, map_location=device)

    # Determine input_dim from the first layer's weight shape
    input_dim = state_dict[list(state_dict.keys())[0]].shape[1]

    if model_type == 'lstm':
        model = LSTMModel(input_dim, **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(input_dim, **model_params)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(state_dict)
    return model

def inference(model, seq):
    """
    Perform inference using the given model and input sequence.

    Args:
        model (torch.nn.Module): The model to use for inference.
        seq (torch.Tensor): Input sequence.

    Returns:
        float: Prediction result.
    """

    with torch.no_grad():
        prediction = model(seq).item()
    return prediction

def ensemble_inference(lstm_model, transformer_model, seq):
    """
    Perform ensemble inference using LSTM and Transformer models.

    Args:
        lstm_model (torch.nn.Module): LSTM model.
        transformer_model (torch.nn.Module): Transformer model.
        seq (torch.Tensor): Input sequence.

    Returns:
        float: Average prediction from both models.
    """

    lstm_pred = inference(lstm_model, seq)
    transformer_pred = inference(transformer_model, seq)
    return (lstm_pred + transformer_pred) / 2

# Trading function
def trade(lstm_model, transformer_model, data, scaler, symbol, initial_balance=100000):
    """
    Simulate trading based on model predictions.

    Args:
        lstm_model (torch.nn.Module): LSTM model.
        transformer_model (torch.nn.Module): Transformer model.
        data (pd.DataFrame): Historical price data.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used for data normalization.
        symbol (str): Stock symbol.
        initial_balance (float): Initial trading balance.

    Returns:
        list: List of returns for each trading day.
    """

    lstm_model.eval()
    transformer_model.eval()

    # Force CPU
    lstm_model = lstm_model.cpu()
    transformer_model = transformer_model.cpu()

    balance = initial_balance
    position = 0
    returns = []


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pbar = tqdm(range(len(data) - 60), desc=f'[{datetime.now().strftime("%H:%M:%S")}] Trading {symbol}')

    with torch.no_grad():
        for i in pbar:
            seq = torch.FloatTensor(data.iloc[i:i+60].values).unsqueeze(0).to(device)
            prediction = ensemble_inference(lstm_model, transformer_model, seq)

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

def process_stock(symbol, start_date, end_date, seq_length):
    """
    Process a single stock for trading simulation.

    Args:
        symbol (str): Stock symbol.
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        seq_length (int): Sequence length for model input.

    Returns:
        tuple: Symbol, total return, and Sharpe ratio.
    """

    try:
        X, y, scaler, data = get_data(symbol, start_date, end_date)


        # Convert all columns to numeric, replacing non-numeric values with NaN
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # Drop any rows with NaN values
        data = data.dropna()

        # Recalculate X and y after cleaning the data
        features = [col for col in data.columns if col != 'returns']
        X = data[features].values
        y = data['returns'].values

        print(f"X shape: {X.shape}, X dtype: {X.dtype}")
        print(f"y shape: {y.shape}, y dtype: {y.dtype}")




        lstm_model = load_model('lstm', symbol, LSTM_PARAMS)
        transformer_model = load_model('transformer', symbol, TRANSFORMER_PARAMS)

        # Simulate trading using ensemble predictions
        returns = trade(lstm_model, transformer_model, data, scaler, symbol)

        returns_array = np.array(returns)
        total_return = returns[-1]
        sharpe_ratio = returns_array.mean() / returns_array.std() * (252 ** 0.5)  # Annualized Sharpe Ratio

        return symbol, total_return, sharpe_ratio

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")
        return symbol, None, None

# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    multiprocessing.set_start_method('spawn', force=True)

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)  # Use last 30 days for inference

    with multiprocessing.Pool(processes=1) as pool:
        results = pool.map(partial(process_stock,
                                   start_date=start_date,
                                   end_date=end_date,
                                   seq_length=TRAINING_PARAMS['seq_length']),
                           SYMBOLS)

    for symbol, total_return, sharpe_ratio in results:
        if total_return is not None and sharpe_ratio is not None:
            print(f"{symbol}:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            print(f"{symbol}: Error during inference")
        print()
