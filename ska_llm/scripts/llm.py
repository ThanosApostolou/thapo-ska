from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import os

import torch

from scripts import rag
from scripts import constants

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


def read_docs(data_path: str):
    raw_text_pickle = "raw_text.pkl"
    raw_text: str = ""

    if os.path.isfile(raw_text_pickle):
        with open(raw_text_pickle, "rb") as f:
            raw_text = pickle.load(f)
    else:
        docs: list[Document] = rag.read_docs(data_path)
        # vocab_set: set[str] = set()
        for doc in docs:
            raw_text = raw_text + doc.page_content.strip().lower()
            # doc_chars = set(doc.page_content.lower())
            # vocab_set = vocab_set.union(doc_chars)

        with open(raw_text_pickle, 'wb') as f:
            pickle.dump(raw_text, f)

    text_tokens = tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)
    text_tokens_len = len(text_tokens)
    vocab = sorted(list(set(text_tokens)))
    vocab_len = len(vocab)

    print('text_tokens', text_tokens[:20])
    print('text_tokens_len', text_tokens_len)
    print('vocab', vocab[:20])
    print('vocab_len', vocab_len)
    return text_tokens, text_tokens_len, vocab, vocab_len


def create_vocab_dicts(vocab: list[str]):
    char_to_int_dict: dict[str, int] = {}
    int_to_char_dict: dict[int, str] = {}
    for i, c in enumerate(vocab):
        char_to_int_dict[c] = i
        int_to_char_dict[i] = c

    print('char_to_int_dict', list(char_to_int_dict.items())[:20])
    print('int_to_char_dict', list(int_to_char_dict.items())[:20])
    return (char_to_int_dict, int_to_char_dict)


def create_thapollm(data_path: str):
    print("create_thapollm start")
    (text_tokens, text_tokens_len, vocab, vocab_len) = read_docs(data_path)
    (char_to_int_dict, int_to_char_dict) = create_vocab_dicts(vocab)

    seq_length = 100
    dataX: list[list[int]] = []
    dataY: list[int] = []
    for i in range(0, text_tokens_len - seq_length, 1):
        seq_in = text_tokens[i:i + seq_length]
        seq_out = text_tokens[i + seq_length]
        dataX.append([char_to_int_dict[char] for char in seq_in])
        dataY.append(char_to_int_dict[seq_out])

    print('dataX', list(map(lambda row: row[:20], dataX[:20])))
    print('dataY', dataY[:20])

    n_patterns = len(dataX)
    X = torch.tensor(dataX, dtype=torch.float32)
    print('X.shape', X.shape)
    X = X.reshape(n_patterns, seq_length, 1)
    print('X.shape', X.shape)
    X = X / float(vocab_len)
    y = torch.tensor(dataY)
    print(X.shape, y.shape)
    dataX_len = len(dataX)
