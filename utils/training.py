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
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from datetime import datetime, timedelta
import multiprocessing
from functools import partial
import logging
from tqdm import tqdm
import warnings
import os
import dotenv
import plotext as plt
import json

from data_processing import get_data
from dataset import StockDataset
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from config import SYMBOLS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="ta")

# Training function
def train_model(X, y, seq_length, model_params, lr, num_epochs, model_type='lstm', device=None):
    """
    Train a model (LSTM or Transformer) on the given data.
    """

    if device is None:
        raise ValueError("Must select device type")

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_dataset = StockDataset(X_train, y_train, seq_length)
        val_dataset = StockDataset(X_val, y_val, seq_length)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        if model_type == 'lstm':
            model = LSTMModel(X.shape[1], **model_params)
        elif model_type == 'transformer':
            model = TransformerModel(X.shape[1], **model_params)
        else:
            raise ValueError("Invalid model type")
        model.to(device)
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # type: ignore

        # One Cycle Learning Rate Scheduler
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader))
        scaler = GradScaler()
        patience = 10
        patience_counter = 0
        fold_best_val_loss = float('inf')
        fold_train_losses = []
        fold_val_losses = []
        pbar = tqdm(range(num_epochs), desc=f'Training {model_type.upper()} model (Fold {fold+1}/5)')
        for epoch in pbar:
            model.train()
            total_train_loss = 0
            for seq, target in train_loader:
                seq, target = seq.to(device), target.to(device)
                optimizer.zero_grad()
                with autocast():
                    output = model(seq)
                    loss = criterion(output.squeeze(), target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            fold_train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(device), target.to(device)
                    output = model(seq)
                    loss = criterion(output.squeeze(), target)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            fold_val_losses.append(avg_val_loss)
            pbar.set_postfix({'Train Loss': f'{avg_train_loss:.4f}', 'Val Loss': f'{avg_val_loss:.4f}'})

            # Early stopping and best model selection
            if avg_val_loss < fold_best_val_loss:
                fold_best_val_loss = avg_val_loss
                patience_counter = 0
                if fold_best_val_loss < best_val_loss:
                    best_val_loss = fold_best_val_loss
                    best_model = model.state_dict()
                    train_losses = fold_train_losses
                    val_losses = fold_val_losses
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
    if best_model is not None:
        model.load_state_dict(best_model) # type: ignore
    return model, best_val_loss, train_losses, val_losses # type: ignore

def plot_loss_curves(train_losses, val_losses, model_type):
    plt.clear_figure()
    plt.plot_size(width=60, height=20)
    plt.theme('dark')

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.title(f'{model_type.upper()} Model - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def hyperparameter_tuning(X, y, seq_length, model_type, device):
    if model_type == 'lstm':
        param_grid = {
            #'hidden_dim': [64, 128, 256],
            'hidden_dim': [64],
            #'num_layers': [1, 2, 3],
            'num_layers': [1],
            #'lr': [0.001, 0.0001],
            'lr': [0.001],
            #'num_epochs': [50, 100, 200]
            'num_epochs': [50]
        }
    elif model_type == 'transformer':
        param_grid = {
            #'d_model': [64, 128, 256],
            'd_model': [64],
            #'nhead': [4, 8],
            'nhead': [4],
            #'num_layers': [1, 2, 3],
            'num_layers': [1],
            #'lr': [0.001, 0.0001],
            'lr': [0.001],
            #'num_epochs': [50, 100, 200]
            'num_epochs': [50]
        }
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    best_model = None
    best_val_loss = float('inf')
    best_params = None
    best_train_losses = None
    best_val_losses = None
    for params in ParameterGrid(param_grid):
        if model_type == 'lstm':
            model_params = {
                'hidden_dim': params['hidden_dim'],
                'num_layers': params['num_layers'],
                'output_dim': 1
            }
        else:  # transformer
            model_params = {
                'd_model': params['d_model'],
                'nhead': params['nhead'],
                'num_layers': params['num_layers'],
                'output_dim': 1
            }
        logging.info(f"Training {model_type} model with parameters: {params}")
        model, val_loss, train_losses, val_losses  = train_model(
            X, y, seq_length, model_params,
            lr=params['lr'],
            num_epochs=params['num_epochs'],
            model_type=model_type,
            device=device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_params = params
            best_train_losses = train_losses
            best_val_losses = val_losses

    print(f"Best parameters: {best_params}") # type: ignore
    print(f"Best validation loss: {best_val_loss:.4f}")
    plot_loss_curves(best_train_losses, best_val_losses, model_type)
    return best_model, best_params  # type: ignore

# Save model to disc
def save_model(model, model_type, symbol, params):
    """
    Save the trained model to disk.
    """

    directory = f"./trained_models/{symbol}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{symbol}_{model_type}_model.pth"
    file_path = os.path.join(directory, file_name)
    torch.save(model.state_dict(), file_path)
    print(f"Saving model: {file_path}")

    info = {
        "model_type": model_type,
        "symbol": symbol,
        "params": params,
        "timestamp": timestamp
    }
    info_filename = f"{symbol}_{model_type}_info.json"
    info_path = os.path.join(directory, info_filename)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Saving model: {info_path}")


# Parallel processing for multiple stocks
def orchestrate_training(symbol, start_date, end_date, seq_length):
    """
    Process a single stock: fetch data, train models, and save them.
    """

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        X, y, scaler, data = get_data(symbol, start_date, end_date)

        # Train and tune LSTM model
        lstm_model, lstm_params = hyperparameter_tuning(X, y, seq_length, 'lstm', device)
        save_model(lstm_model, 'lstm', symbol, lstm_params)
        logging.info(f"Best LSTM parameters for {symbol}: {lstm_params}")

        # Train and tune Transformer model
        transformer_model, transformer_params = hyperparameter_tuning(X, y, seq_length, 'transformer', device)
        save_model(transformer_model, 'transformer', symbol, transformer_params)
        logging.info(f"Best Transformer parameters for {symbol}: {transformer_params}")

        logging.info(f"Successfully processed {symbol}")

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")


# Main execution
if __name__ == "__main__":

    # Set start method to spawn
    multiprocessing.set_start_method('spawn', force=True)

    end_date = datetime.now() - timedelta(days=1)  # Use yesterday's date
    start_date = end_date - timedelta(days=10)  # Use one year of data

    seq_length = 60

    # Parallel processing
    with multiprocessing.Pool() as pool:
        pool.map(partial(orchestrate_training, start_date=start_date, end_date=end_date,
                        seq_length=seq_length), SYMBOLS)
