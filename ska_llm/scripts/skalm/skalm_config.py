class SkalmConfig:
    def __init__(self) -> None:
        self.seq_len = 100
        self.embedding_dim = 64
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 1
        self.dropout = 0.1

        # train
        self.n_epochs = 4
        self.batch_size = 16

        # invoke
        self.max_tokens = 40
        self.max_eos = 5