from typing import Any
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from ska_llm.scripts.skalm.skalm_config import SkalmConfig

class Skalm(nn.Module):
    def __init__(self, skalm_config: SkalmConfig, n_vocab: int):
        super(Skalm, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=skalm_config.embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(input_size=skalm_config.embedding_dim,
                            hidden_size=skalm_config.lstm_hidden_size,
                            num_layers=skalm_config.lstm_num_layers,
                            batch_first=True,
                            dropout=skalm_config.dropout)
        # self.dropout = nn.Dropout(skalm_config.dropout)
        self.linear = nn.Linear(skalm_config.lstm_hidden_size, n_vocab)


    def forward(self, x):
        # embedding
        x = self.embedding(x)

        # lstm
        x, _ = self.lstm(x)

        # take only the last output
        x = x[:, -1, :]

        # dropout
        # x = self.dropout(x)

        # produce output
        x = self.linear(x)
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


def plot_losses(skalm_dir_path: str, train_losses: list[float], title: str):
    epoch_list = [i for i in range(len(train_losses))]
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(epoch_list, train_losses, '-r', label='Cross-Entropy Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(f"{skalm_dir_path}/{title}.png")


def save_model(skalm_dir_path: str, best_model_state_dict: dict[str, Any] | None, train_losses: list[float], test_losses: list[float], epoch: int | None):
    suffix = f"_{epoch}" if epoch is not None else ""
    model_path = f"{skalm_dir_path}/skalm{suffix}.pth"
    train_title = f"skalm_train_loss{suffix}"
    plot_losses(skalm_dir_path, train_losses, train_title)
    test_title = f"skalm_test_loss{suffix}"
    plot_losses(skalm_dir_path, test_losses, test_title)
    torch.save(best_model_state_dict, model_path)


def train_skallm_lstm(model: Skalm, skalm_config: SkalmConfig, skalm_dir_path: str, Xtrain: Tensor, Ytrain: Tensor, Xtest: Tensor, Ytest: Tensor) -> tuple[dict[str, Any] | None, list[float], list[float]]:
    print("train_skallm_lstm start")
    n_epochs = skalm_config.n_epochs
    batch_size = skalm_config.batch_size

    optimizer = optim.Adam(model.parameters(), lr=skalm_config.lr)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    # dataset = TensorDataset(X, y)
    # dataset_train, dataset_test = random_split(dataset, [0.8, 0.2])
    dataset_train = TensorDataset(Xtrain, Ytrain)
    dataset_test = TensorDataset(Xtest, Ytest)
    print('len(dataset_train)', len(dataset_train))
    print('len(dataset_test)', len(dataset_test))

    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    loader_test = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

    best_model_state_dict = None
    best_loss = np.inf
    train_losses: list[float] = []
    test_losses: list[float] = []
    for epoch in range(n_epochs):
        model.train(True)
        print(f"Epoch {epoch}: train_one_epoch")
        train_avg_loss = train_one_epoch(model, loader_train, optimizer, loss_fn)
        train_losses.append(train_avg_loss)
        print("Epoch %d: train_loss: %.4f" % (epoch, train_avg_loss))

        # Validation
        print("Epoch %d: validate model" % (epoch,))
        model.eval()
        with torch.no_grad():
            test_total_loss = 0
            for X_batch, y_batch in loader_test:
                y_pred = model(X_batch)
                test_loss = loss_fn(y_pred, y_batch)
                test_total_loss += test_loss.item()

            test_avg_loss = test_total_loss / len(loader_test)
            test_losses.append(test_avg_loss)
            if test_avg_loss <= best_loss:
                best_loss = test_avg_loss
                best_model_state_dict = model.state_dict()
                save_model(skalm_dir_path, best_model_state_dict, train_losses, test_losses, epoch)

            print("Epoch %d: test_loss: %.4f" % (epoch, test_avg_loss))


    save_model(skalm_dir_path, best_model_state_dict, train_losses, test_losses, None)
    print("train_skallm_lstm end")
    return best_model_state_dict, train_losses, test_losses


def predict(model: Skalm, encoded_tokens: Tensor) -> Tensor:
    pred_logits = model(encoded_tokens)
    print("pred_logits: ", len(pred_logits), pred_logits)
    predicted_labels = torch.argmax(pred_logits, dim=1)
    print("predicted_labels: ", len(predicted_labels), predicted_labels)
    return predicted_labels