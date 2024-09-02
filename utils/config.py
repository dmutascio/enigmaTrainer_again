"""
This module contains configuration parameters for the stock trading models.

It includes parameters for LSTM and Transformer models, training settings,
and the list of stock symbols to process.
"""

LSTM_PARAMS = {
    'hidden_dim': 128,
    'num_layers': 3,
    'output_dim': 1
}

TRANSFORMER_PARAMS = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 3,
    'output_dim': 1
}

TRAINING_PARAMS = {
    'seq_length': 60,
    'lr': 0.001,
    'num_epochs': 100
}

SYMBOLS = ['AAPL']
#SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
