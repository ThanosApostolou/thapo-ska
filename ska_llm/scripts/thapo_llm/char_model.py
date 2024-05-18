import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class CharModel(nn.Module):
    def __init__(self, n_vocab: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

def train_char_model(n_vocab: int, char_to_int: dict[str, int], X: Tensor, y: Tensor):
    n_epochs = 40
    batch_size = 128
    model = CharModel(n_vocab=n_vocab)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

    best_model = None
    best_loss = np.inf
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

    torch.save([best_model, char_to_int], "single-char.pth")