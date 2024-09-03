"""
This module handles the training of LSTM and Transformer models.

It includes model definitions, training functions, and utilities for
processing multiple stocks in parallel.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime, timedelta
import multiprocessing
from functools import partial
import logging
from tqdm import tqdm
import warnings
import os
import dotenv

from data_processing import get_data
from dataset import StockDataset
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", message="A NumPy version")
warnings.filterwarnings("ignore", message="Pandas requires version")

# Training function
def train_model(X, y, seq_length, model_params, lr, num_epochs, model_type='lstm', device=None):
    """
    Train a model (LSTM or Transformer) on the given data.

    Args:
        X (np.array): Input features.
        y (np.array): Target values.
        seq_length (int): Sequence length for the model input.
        model_params (dict): Parameters for the model.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        model_type (str): Type of model to train ('lstm' or 'transformer').
        device (torch.device): Device to run the model on.

    Returns:
        torch.nn.Module: Trained model.

    Raises:
        ValueError: If an invalid model type is provided or no device is specified.
    """

    if device is None:
        raise ValueError("Must select device type")

    dataset = StockDataset(np.column_stack((y.reshape(-1, 1), X)), seq_length)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    if model_type == 'lstm':
        model = LSTMModel(X.shape[1] + 1, **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(X.shape[1] + 1, **model_params)
    else:
        raise ValueError("Invalid model type")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    pbar = tqdm(range(num_epochs), desc=f'Training {model_type.upper()} model')
    for epoch in pbar:
        model.train()
        total_loss = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(seq)
                loss = criterion(output.squeeze(), target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

    return model

# Save model to disc
def save_model(model, model_type, symbol):
    """
    Save the trained model to disk.

    Args:
        model (torch.nn.Module): The trained model to save.
        model_type (str): Type of the model ('lstm' or 'transformer').
        symbol (str): Stock symbol.
    """

    directory = f"../trained_models/{symbol}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = f"{directory}/{symbol}_{model_type}_model.pth"
    print(f"Saving model: {file_path}")
    torch.save(model.state_dict(), file_path)

# Parallel processing for multiple stocks
def process_stock(symbol, start_date, end_date, seq_length, lstm_params, transformer_params, lr, num_epochs):
    """
    Process a single stock: fetch data, train models, and save them.

    Args:
        symbol (str): Stock symbol.
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        seq_length (int): Sequence length for model input.
        lstm_params (dict): Parameters for the LSTM model.
        transformer_params (dict): Parameters for the Transformer model.
        lr (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
    """

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        X, y, scaler, data = get_data(symbol, start_date, end_date)

        lstm_model = train_model(X, y, seq_length, lstm_params, lr, num_epochs, 'lstm', device)
        transformer_model = train_model(X, y, seq_length, transformer_params, lr, num_epochs, 'transformer', device)

        save_model(lstm_model, 'lstm', symbol)
        save_model(transformer_model, 'transformer', symbol)

        logging.info(f"Successfully processed {symbol}")

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")


# Main execution
if __name__ == "__main__":

    # Set start method to spawn
    multiprocessing.set_start_method('spawn', force=True)

    #symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # Stocks
    symbols = ['AAPL']  # Stocks TESTING

    end_date = datetime.now() - timedelta(days=1)  # Use yesterday's date
    start_date = end_date - timedelta(days=365)  # Use one year of data

    seq_length = 60
    lstm_params = {'hidden_dim': 128, 'num_layers': 3, 'output_dim': 1}
    transformer_params = {'d_model': 128, 'nhead': 8, 'num_layers': 3, 'output_dim': 1}
    lr = 0.001
    num_epochs = 100

    # Parallel processing
    with multiprocessing.Pool(processes=1) as pool:  # Use only 1 process for testing
        pool.map(partial(process_stock, start_date=start_date, end_date=end_date,
                                   seq_length=seq_length, lstm_params=lstm_params,
                                   transformer_params=transformer_params, lr=lr, num_epochs=num_epochs), symbols)
