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
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotext as plt

dotenv.load_dotenv()

# Initialize Alpaca clients
API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def print_returns_graph(daily_returns, symbol):
    # Convert daily returns to numpy array if it's not already
    daily_returns = np.array(daily_returns)

    # Calculate cumulative returns
    cumulative_returns = np.cumsum(daily_returns)

    # Add initial 0 and divide by 100
    cumulative_returns = np.insert(cumulative_returns, 0, 0) / 100

    # Create x-axis (days)
    bars = list(range(len(cumulative_returns)))

    # Clear any previous plot
    plt.clear_figure()

    plt.plot_size(width=60, height=15)

    plt.theme('dark')

    # Create the plot
    plt.plot(bars, cumulative_returns, label="Cumulative Returns", color="green")
    plt.title(f"Cumulative Returns for {symbol}")
    plt.xlabel("bars")
    plt.ylabel("Returns")

    # Show the plot
    plt.show()

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

    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the trained_models directory
    trained_models_dir = os.path.join(current_script_dir, "..", "trained_models")

    # Construct the full path to the model file
    file_path = os.path.join(trained_models_dir, symbol, f"{symbol}_{model_type}_model.pth")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

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
    model.eval()  # Set the model to evaluation mode
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
    # Force CPU
    lstm_model = lstm_model.cpu()
    transformer_model = transformer_model.cpu()

    lstm_model.eval()
    transformer_model.eval()

    balance = initial_balance
    position = 0
    returns = []

    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")

    pbar = tqdm(range(len(data) - 60), desc=f'[{datetime.now().strftime("%H:%M:%S")}] Trading {symbol}')

    with torch.no_grad():
        for i in pbar:
            seq = torch.FloatTensor(data[i:i+60]).unsqueeze(0).to(device)
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

def simulate_trading(symbol, start_date, end_date, lstm_model, transformer_model, seq_length, X, y, scaler, data, initial_balance=100000, transaction_cost=0.001):
    """
    Simulate trading using ensemble predictions from LSTM and Transformer models.
    """

    balance = initial_balance
    position = 0
    trades = []
    daily_returns = []
    predictions = []
    actual_returns = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)
    transformer_model.to(device)

    pbar = tqdm(range(seq_length, len(X)), desc=f'[{datetime.now().strftime("%H:%M:%S")}] Simulating {symbol}')

    for i in pbar:
        seq = torch.FloatTensor(X[i-seq_length:i]).unsqueeze(0).to(device)

        prediction = ensemble_inference(lstm_model, transformer_model, seq)

        predictions.append(prediction)
        actual_returns.append(y[i])

        current_price = data.iloc[i]['close']

        if prediction > 0.001 and position == 0:  # Buy signal
            shares_to_buy = balance // current_price
            cost = shares_to_buy * current_price * (1 + transaction_cost)
            if cost <= balance:
                position = shares_to_buy
                balance -= cost
                trades.append(('BUY', i, current_price, shares_to_buy, cost))
        elif prediction < -0.001 and position > 0:  # Sell signal
            revenue = position * current_price * (1 - transaction_cost)
            balance += revenue
            trades.append(('SELL', i, current_price, position, revenue))
            position = 0

        total_value = balance + position * current_price
        daily_return = (total_value / initial_balance) - 1
        daily_returns.append(daily_return)

        pbar.set_postfix({'Returns': f'{daily_return:.2%}'})

    final_balance = balance + position * data.iloc[-1]['close']
    total_return = (final_balance / initial_balance) - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized

    threshold = 0.001
    y_true = [1 if ret > threshold else (0 if ret < -threshold else 2) for ret in actual_returns]
    y_pred = [1 if pred > threshold else (0 if pred < -threshold else 2) for pred in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    results = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'number_of_trades': len(trades),
        'trades': trades,
        'daily_returns': daily_returns,
        'predictions': predictions,
        'actual_returns': actual_returns
    }

    return results


def process_stock(symbol, start_date, end_date, seq_length, simulate):
    """
    Process a single stock for trading simulation.

    Args:
        symbol (str): Stock symbol.
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        seq_length (int): Sequence length for model input.
        live_mode (boolean): Live or simulation mode


    Returns:
        tuple: Symbol, total return, and Sharpe ratio.
    """

    try:
        X, y, scaler, data = get_data(symbol, start_date, end_date)

        lstm_model = load_model('lstm', symbol, LSTM_PARAMS)
        transformer_model = load_model('transformer', symbol, TRANSFORMER_PARAMS)

        if simulate:
            # Perform live trading
            results = simulate_trading(symbol, start_date, end_date, lstm_model, transformer_model, seq_length, X, y, scaler, data)

        else:
            # Simulate
            results = trade(lstm_model, transformer_model, X, scaler, symbol)

        return symbol, results

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")
        return symbol, None

# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run trading simulation or live trading")
    parser.add_argument("--simulate", default=True)
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=25)  # Use last 30 days for inference
    #start_date = end_date - timedelta(days=30)  # Use last 30 days for inference

    # Parallel processing
    with multiprocessing.Pool() as pool:
        results = pool.map(partial(process_stock,
                                   start_date=start_date,
                                   end_date=end_date,
                                   seq_length=TRAINING_PARAMS['seq_length'],
                                   simulate=args.simulate),
                           SYMBOLS)

    for result in results:
        symbol, data = result # type: ignore
        if data is not None:
            print(f"\nResults for {symbol}:")
            print(f"Total Return: {data['total_return']:.2%}") # type: ignore
            print(f"Sharpe Ratio: {data['sharpe_ratio']:.2f}") # type: ignore
            print(f"Accuracy: {data['accuracy']:.2f}") # type: ignore
            print(f"Precision: {data['precision']:.2f}") # type: ignore
            print(f"Recall: {data['recall']:.2f}") # type: ignore
            print(f"F1 Score: {data['f1_score']:.2f}") # type: ignore
            print(f"Number of Trades: {data['number_of_trades']}") # type: ignore

            # Print the graph
            print_returns_graph(data['daily_returns'], symbol) # type: ignore
        else:
            print(f"{symbol}: Error during processing")
        print()
