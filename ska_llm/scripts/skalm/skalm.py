from typing import Any
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from ska_llm.scripts.skalm.skalm_config import SkalmConfig, SkalmProps

class Skalm(nn.Module):
    def __init__(self, skalm_props: SkalmProps, n_vocab: int):
        super(Skalm, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=skalm_props.embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(input_size=skalm_props.embedding_dim,
                            hidden_size=skalm_props.lstm_hidden_size,
                            num_layers=skalm_props.lstm_num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(skalm_props.dropout)
        self.linear = nn.Linear(skalm_props.lstm_hidden_size, n_vocab)


    def forward(self, x):
        # embedding
        x = self.embedding(x)

        # lstm
        x, _ = self.lstm(x)

        # dropout
        x = self.dropout(x)

        # take only the last output
        x = x[:, -1, :]

        # produce output
        x = self.linear(x)
        return x



def train_one_epoch(model: Skalm, loader: DataLoader, optimizer: optim.Adam, loss_fn: nn.CrossEntropyLoss) -> tuple[float, float]:
    train_epoch_total_loss = 0
    train_epoch_total_correct = 0
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

        # calculate
        train_epoch_total_loss += loss.item()
        winners = y_pred.argmax(dim=1)
        train_epoch_total_correct += (y_batch == winners).sum().item() / len(y_batch)

    train_epoch_loss = train_epoch_total_loss / len(loader)
    train_epoch_accuracy = train_epoch_total_correct / len(loader)
    return train_epoch_loss, train_epoch_accuracy


def validate_one_epoch(model: Skalm, loader_test: DataLoader, loss_fn: nn.CrossEntropyLoss):
    test_epoch_total_loss = 0
    test_epoch_total_correct = 0
    for X_batch, y_batch in loader_test:
        y_pred = model(X_batch)
        test_loss = loss_fn(y_pred, y_batch)
        test_epoch_total_loss += test_loss.item()
        winners = y_pred.argmax(dim=1)
        test_epoch_total_correct += (y_batch == winners).sum().item() / len(y_batch)

    test_epoch_loss = test_epoch_total_loss / len(loader_test)
    test_epoch_accuracy = test_epoch_total_correct / len(loader_test)
    return test_epoch_loss, test_epoch_accuracy


def plot_losses(skalm_dir_path: str, train_losses: list[float], title: str, ylabel: str):
    epoch_list = [i for i in range(len(train_losses))]
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(epoch_list, train_losses, '-r', label=ylabel)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(f"{skalm_dir_path}/{title}.png")


def save_model(skalm_dir_path: str, best_model_state_dict: dict[str, Any] | None, train_losses: list[float], train_accuracy_list: list[float], test_losses: list[float], test_accuracy_list: list[float], epoch: int | None):
    suffix = f"_{epoch}" if epoch is not None else ""
    model_path = f"{skalm_dir_path}/skalm{suffix}.pth"
    loss_ylabel = "Cross-Entropy Loss"
    accuracy_ylabel = "Accuracy"
    train_loss_title = f"skalm_train_loss{suffix}"
    plot_losses(skalm_dir_path, train_losses, train_loss_title, loss_ylabel)
    train_accuracy_title = f"skalm_train_accuracy{suffix}"
    plot_losses(skalm_dir_path, train_accuracy_list, train_accuracy_title, accuracy_ylabel)
    test_loss_title = f"skalm_test_loss{suffix}"
    plot_losses(skalm_dir_path, test_losses, test_loss_title, loss_ylabel)
    test_accuracy_title = f"skalm_test_accuracy{suffix}"
    plot_losses(skalm_dir_path, test_accuracy_list, test_accuracy_title, accuracy_ylabel)
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
    train_accuracy_list: list[float] = []
    test_losses: list[float] = []
    test_accuracy_list: list[float] = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        model.train(True)
        train_epoch_loss, train_epoch_accuracy = train_one_epoch(model, loader_train, optimizer, loss_fn)
        train_losses.append(train_epoch_loss)
        train_accuracy_list.append(train_epoch_accuracy)
        print("Epoch %d: train_epoch_loss: %.4f" % (epoch, train_epoch_loss))
        print("Epoch %d: train_epoch_accuracy: %.4f" % (epoch, train_epoch_accuracy))

        # Validation
        model.eval()
        with torch.no_grad():
            test_epoch_loss, test_epoch_accuracy = validate_one_epoch(model, loader_test, loss_fn)
            test_losses.append(test_epoch_loss)
            test_accuracy_list.append(test_epoch_accuracy)

            if test_epoch_loss <= best_loss:
                best_loss = test_epoch_loss
                best_model_state_dict = model.state_dict()
                save_model(skalm_dir_path, best_model_state_dict, train_losses, train_accuracy_list, test_losses, test_accuracy_list, epoch)

            print("Epoch %d: test_epoch_loss: %.4f" % (epoch, test_epoch_loss))
            print("Epoch %d: test_epoch_accuracy: %.4f" % (epoch, test_epoch_accuracy))


    save_model(skalm_dir_path, best_model_state_dict, train_losses, train_accuracy_list, test_losses, test_accuracy_list, None)
    print("train_skallm_lstm end")
    return best_model_state_dict, train_losses, test_losses


def predict(model: Skalm, encoded_tokens: Tensor) -> Tensor:
    pred_logits = model(encoded_tokens)
    print("pred_logits: ", len(pred_logits), pred_logits)
    predicted_labels = torch.argmax(pred_logits, dim=1)
    print("predicted_labels: ", len(predicted_labels), predicted_labels)
    return predicted_labels