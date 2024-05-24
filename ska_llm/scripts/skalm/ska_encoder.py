
import json

def create_vocab_dicts(vocab: list[str]):
    token_to_int_dict: dict[str, int] = {}
    int_to_token_dict: dict[int, str] = {}
    for i, c in enumerate(vocab):
        token_to_int_dict[c] = i
        int_to_token_dict[i] = c

    return (token_to_int_dict, int_to_token_dict)

class SkaEncoder:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.vocab_len = len(vocab)
        (self.token_to_int_dict, self.int_to_token_dict) = create_vocab_dicts(self.vocab)

    def encode(self, token: str) -> int:
        return self.token_to_int_dict[token]

    def encode_list(self, tokens: list[str]) -> list[int]:
        return list(map(self.encode, tokens))

    def decode(self, encoded_token: int) -> str:
        return self.int_to_token_dict[encoded_token]

    def decode_list(self, encoded_tokens: list[int]) -> list[str]:
        return list(map(self.decode, encoded_tokens))

    def to_json(self) -> str:
        json_obj = {
            "vocab": self.vocab
        }
        return json.dumps(json_obj)

    @staticmethod
    def from_text_tokens(text_tokens: list[str]) -> 'SkaEncoder':
        vocab = sorted(list(set(text_tokens)))
        return SkaEncoder(vocab=vocab)

    @staticmethod
    def from_json(json_str: str) -> 'SkaEncoder':
        json_obj = json.loads(json_str)
        vocab = json_obj['vocab']
        return SkaEncoder(vocab=vocab)