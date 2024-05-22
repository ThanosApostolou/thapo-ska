from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from char_model import train_char_model

def thapo_llm_flow():
    # load ascii text and covert to lowercase
    filename = "wonderland.txt"
    filepath = Path(__file__).parent.joinpath(filename).resolve()

    raw_text = open(filepath, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    print('chars', chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    print('char_to_int', char_to_int)

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    print()

    # prepare the dataset of input to output pairs encoded as integers
    print("prepare the dataset of input to output pairs encoded as integers")
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    print('dataX[:10]', dataX[:10])
    print('dataY[:10]', dataY[:10])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    print()

    # reshape X to be [samples, time steps, features]
    print("reshape X to be [samples, time steps, features]")
    X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
    X = X / float(n_vocab)
    y = torch.tensor(dataY)
    print(X.shape, y.shape)
    print()

    # train char_model
    print("train char_model")
    train_char_model(n_vocab, char_to_int, X, y)
