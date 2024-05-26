
import json
from scripts import constants
from nltk.tokenize import word_tokenize, sent_tokenize

from ska_llm.scripts.skalm.skalm_config import SkalmConfig


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

    def tokenize_sentences(self, sentences: list[str], skalm_config: SkalmConfig) -> list[str]:

        max_words=0
        tokens: list[str] = []
        for sentence in sentences:
            words: list[str] = self.tokenize_text(sentence, constants.TOKENIZE_METHOD_NLTK_WORD)
            words_len = len(words)
            if words_len > max_words:
                max_words = len(words)

            if words_len >= skalm_config.filter_sentence_len_geq and words_len <= skalm_config.filter_sentence_len_leq:
                words.append(self.token_eos)
                tokens.extend(words)

        print('max_words', max_words)
        return tokens


    def to_json(self) -> str:
        json_obj = {
            "vocab": self.vocab
        }
        return json.dumps(json_obj)



    def save_json_file(self, skalm_dir_path: str):
        with open(f"{skalm_dir_path}/ska_tokenizer.json", "w") as f:
            skalm_tokenizer_json = self.to_json()
            json.dump(skalm_tokenizer_json, f, indent=4)


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

    @staticmethod
    def from_raw_text(raw_text: str) -> 'SkaTokenizer':
        text_tokens = SkaTokenizer.tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)
        text_vocab = sorted(list(set(text_tokens)))
        vocab = [SkaTokenizer.token_pad, SkaTokenizer.token_unk, SkaTokenizer.token_eos]
        vocab.extend(text_vocab)
        return SkaTokenizer(vocab=vocab)


    @staticmethod
    def from_json(json_str: str) -> 'SkaTokenizer':
        json_obj = json.loads(json_str)
        vocab = json_obj['vocab']
        return SkaTokenizer(vocab=vocab)

    @staticmethod
    def from_json_file(skalm_dir_path: str) -> 'SkaTokenizer':
        with open(f"{skalm_dir_path}/ska_tokenizer.json", "r") as f:
            skalm_tokenizer_json = json.load(f)
            return SkaTokenizer.from_json(skalm_tokenizer_json)