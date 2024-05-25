import json
from typing import Any

class SkalmConfig:
    def __init__(self, seq_len: int = 100, embedding_dim: int = 64, lstm_hidden_size: int = 128, lstm_num_layers: int = 1, dropout: float = 0.1,
                 n_epochs: int = 4, batch_size: int = 16, max_tokens: int = 40, max_eos: int = 5) -> None:
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout

        # train
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # invoke
        self.max_tokens = max_tokens
        self.max_eos = max_eos


    @staticmethod
    def from_json(skalm_config_json: dict[Any, Any]) -> 'SkalmConfig':

        return SkalmConfig(seq_len=skalm_config_json["seq_len"],
                            embedding_dim=skalm_config_json["embedding_dim"],
                            lstm_hidden_size=skalm_config_json["lstm_hidden_size"],
                            lstm_num_layers=skalm_config_json["lstm_num_layers"],
                            dropout=skalm_config_json["dropout"],
                            n_epochs=skalm_config_json["n_epochs"],
                            batch_size=skalm_config_json["batch_size"],
                            max_tokens=skalm_config_json["max_tokens"],
                            max_eos=skalm_config_json["max_eos"])

    @staticmethod
    def from_json_file(skalm_config_path: str) -> 'SkalmConfig':
        with open(f"{skalm_config_path}", "r") as f:
            skalm_config_json = json.load(f)
            return SkalmConfig.from_json(skalm_config_json)