from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import pickle
import os
from nltk.tokenize import word_tokenize, sent_tokenize

import torch

from scripts import rag
from scripts import constants
from scripts.skalm.skalm import Skalm, train_skallm_lstm
from scripts.skalm.skalm_config import SkalmConfig
from ska_llm.scripts.skalm.ska_encoder import SkaEncoder

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

    return raw_text


def create_thapollm(data_path: str):
    print("create_thapollm start")
    skalm_config = SkalmConfig()
    raw_text = read_docs(data_path)

    text_tokens = tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)

    ska_encoder = SkaEncoder.from_text_tokens(text_tokens)

    encoded_tokens = ska_encoder.encode_list(text_tokens)
    encoded_tokens_len = len(encoded_tokens)

    seq_len = skalm_config.seq_len
    dataX: list[list[int]] = []
    dataY: list[int] = []
    for i in range(0, encoded_tokens_len - seq_len, 1):
        seq_in = encoded_tokens[i:i + seq_len]
        seq_out = encoded_tokens[i + seq_len]
        dataX.append(seq_in)
        dataY.append(seq_out)

    print('dataX', list(map(lambda row: row[:20], dataX[:20])))
    print('dataY', dataY[:20])

    n_patterns = len(dataX)
    X = torch.tensor(dataX, dtype=torch.float32)
    print('X.shape', X.shape)
    X = X.reshape(n_patterns, seq_len, 1)
    print('X.shape', X.shape)
    X = X / float(ska_encoder.vocab_len)
    y = torch.tensor(dataY)
    print(X.shape, y.shape)
    dataX_len = len(dataX)



    model = Skalm(skalm_config, n_vocab=ska_encoder.vocab_len)
    best_model_state_dict = train_skallm_lstm(model, skalm_config, X, y)

    torch.save(best_model_state_dict, "single-char.pth")
