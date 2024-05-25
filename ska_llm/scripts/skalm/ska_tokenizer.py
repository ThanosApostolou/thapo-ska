
import json
from scripts import constants
from nltk.tokenize import word_tokenize, sent_tokenize


class SkaTokenizer:
    token_pad = "<!@#$%^&*_SKA_PAD_*&^%$#@!>"
    token_unk = "<!@#$%^&*_SKA_UNK_*&^%$#@!>"
    token_eos = "<!@#$%^&*_SKA_EOS_*&^%$#@!>"
    encoded_token_pad = 0
    encoded_token_unk = 1
    encoded_token_eos = 2

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.vocab_len = len(vocab)
        (self.token_to_int_dict, self.int_to_token_dict) = self._create_vocab_dicts(self.vocab)


    def _create_vocab_dicts(self, vocab: list[str]):
        token_to_int_dict: dict[str, int] = {
        }
        int_to_token_dict: dict[int, str] = {}
        for i, c in enumerate(vocab):
            token_to_int_dict[c] = i
            int_to_token_dict[i] = c

        return (token_to_int_dict, int_to_token_dict)


    def encode(self, token: str) -> int:
        encoded_token = self.token_to_int_dict[token] if token in self.token_to_int_dict else self.encoded_token_unk
        return encoded_token


    def encode_list(self, tokens: list[str]) -> list[int]:
        return list(map(self.encode, tokens))


    def decode(self, encoded_token: int) -> str:
        return self.int_to_token_dict[encoded_token]


    def decode_list(self, encoded_tokens: list[int]) -> list[str]:
        return list(map(self.decode, encoded_tokens))

    def tokenize_raw_text(self, raw_text: str) -> list[str]:
        sentences: list[str] = self.tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_SENT)

        tokens: list[str] = []
        for sentence in sentences:
            words: list[str] = self.tokenize_text(sentence, constants.TOKENIZE_METHOD_NLTK_WORD)
            words.append(self.token_eos)
            tokens.extend(words)

        return tokens


    def to_json(self) -> str:
        json_obj = {
            "vocab": self.vocab
        }
        return json.dumps(json_obj)


    @staticmethod
    def tokenize_text(text: str, method: str) -> list[str]:
        tokens: list[str] = []
        if constants.TOKENIZE_METHOD_CHAR == method:
            tokens = [char for char in text]
        elif constants.TOKENIZE_METHOD_NLTK_WORD == method:
            tokens = word_tokenize(text)
        elif constants.TOKENIZE_METHOD_NLTK_SENT == method:
            tokens = sent_tokenize(text)
        else:
            raise Exception(f"tokenize_text unsuported method {method}")

        return tokens

    @classmethod
    def from_raw_text(cls, raw_text: str) -> 'SkaTokenizer':
        text_tokens = cls.tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)
        text_vocab = sorted(list(set(text_tokens)))
        vocab = [cls.token_pad, cls.token_unk, cls.token_eos]
        vocab.extend(text_vocab)
        return cls(vocab=vocab)


    @staticmethod
    def from_json(json_str: str) -> 'SkaTokenizer':
        json_obj = json.loads(json_str)
        vocab = json_obj['vocab']
        return SkaTokenizer(vocab=vocab)