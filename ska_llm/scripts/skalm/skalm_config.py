class SkalmConfig:
    def __init__(self) -> None:
        self.seq_len = 100
        self.lstm_hidden_size = 128
        self.dropout = 0.2
        self.n_epochs = 20
        self.batch_size = 64