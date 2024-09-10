"""
This module defines the custom PyTorch Dataset for stock data.

It includes the StockDataset class which prepares sequential data
for training the LSTM and Transformer models.
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    """
    A custom Dataset class for preparing stock data for sequential models.

    Attributes:
        data (torch.FloatTensor): The stock data converted to a PyTorch tensor.
        seq_length (int): The length of each sequence in the dataset.
    """

    def __init__(self, X, y, seq_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.seq_length],
            self.y[idx+self.seq_length-1]
        )
