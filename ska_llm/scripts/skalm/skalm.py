from typing import Any
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from ska_llm.scripts.skalm.skalm_config import SkalmConfig

class Skalm(nn.Module):
    def __init__(self, skalm_config: SkalmConfig, n_vocab: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=skalm_config.lstm_hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(skalm_config.dropout)
        self.linear = nn.Linear(skalm_config.lstm_hidden_size, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


def train_one_epoch(model: Skalm, loader: DataLoader, optimizer: optim.Adam, loss_fn: nn.CrossEntropyLoss):
    total_loss = 0
    for X_batch, y_batch in loader:
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_pred = model(X_batch)

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y_batch)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        total_loss += loss.item()

    avg_epoch_loss = total_loss / len(loader)
    return avg_epoch_loss


def train_skallm_lstm(model: Skalm, skalm_config: SkalmConfig, X: Tensor, y: Tensor) -> dict[str, Any] | None:
    print("train_skallm_lstm start")
    n_epochs = skalm_config.n_epochs
    batch_size = skalm_config.batch_size

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    dataset = TensorDataset(X, y)
    dataset_train, dataset_test = random_split(dataset, [0.8, 0.2])
    print('len(dataset_train)', len(dataset_train))
    print('len(dataset_test)', len(dataset_test))

    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    loader_test = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

    best_model_state_dict = None
    best_loss = np.inf
    for epoch in range(n_epochs):
        model.train(True)
        print(f"train_one_epoch epoch: {epoch}")
        avg_epoch_loss = train_one_epoch(model, loader_train, optimizer, loss_fn)

        # Validation
        print(f"validate mode epoch: {epoch}")
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader_test:
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch)
            if loss < best_loss:
                best_loss = loss
                best_model_state_dict = model.state_dict()
            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))


    print("train_skallm_lstm end")
    return best_model_state_dict
