from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import pickle
import os

import torch

from scripts import rag
from scripts import constants
from scripts.skalm.skalm import Skalm, predict, train_skallm_lstm
from scripts.skalm.skalm_config import SkalmConfig
from ska_llm.scripts.skalm.ska_tokenizer import SkaTokenizer




def read_docs(data_path: str, ska_tmp_dir: str):
    raw_text_pickle = f"{ska_tmp_dir}/raw_text.pkl"
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


def create_skalm(data_path: str, skalm_dir_path: str, skalm_config_path: str, ska_tmp_dir: str):
    print("create_thapollm start")
    if not os.path.exists(ska_tmp_dir):
        os.makedirs(ska_tmp_dir, exist_ok=True)

    skalm_config = SkalmConfig.from_json_file(skalm_config_path)
    raw_text = read_docs(data_path, ska_tmp_dir)

    # text_tokens = tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)

    ska_tokenizer = SkaTokenizer.from_raw_text(raw_text)
    seq_len = skalm_config.seq_len
    tokens  = ska_tokenizer.tokenize_raw_text(raw_text)
    encoded_tokens = ska_tokenizer.encode_list(tokens)
    encoded_tokens_len = len(encoded_tokens)

    dataX: list[list[int]] = []
    dataY: list[int] = []
    for i in range(0, encoded_tokens_len - seq_len, 1):
        seq_in = encoded_tokens[i:i + seq_len]
        seq_out = encoded_tokens[i + seq_len]
        dataX.append(seq_in)
        dataY.append(seq_out)

    print('seq_in, seq_len',len(seq_in), seq_len)
    print('dataX', list(map(lambda row: row[:20], dataX[:20])))
    print('dataY', dataY[:20])

    n_patterns = len(dataX)
    X = torch.tensor(dataX, dtype=torch.int64)
    print('X.shape', X.shape)
    X = X.reshape(n_patterns, seq_len)
    print('X.shape', X.shape)
    # X = X / float(ska_tokenizer.vocab_len)
    y = torch.tensor(dataY)
    print(X.shape, y.shape)

    ska_tokenizer.save_json_file(skalm_dir_path)
    model = Skalm(skalm_config, n_vocab=ska_tokenizer.vocab_len)
    best_model_state_dict, train_losses = train_skallm_lstm(model, skalm_config, skalm_dir_path, X, y)



def invoke_skalm(question: str, skalm_dir_path: str, skalm_config_path: str) -> str:
    skalm_config = SkalmConfig.from_json_file(skalm_config_path)
    ska_tokenizer = SkaTokenizer.from_json_file(skalm_dir_path)
    model = Skalm(skalm_config, n_vocab=ska_tokenizer.vocab_len)
    model.load_state_dict(torch.load("single-char.pth"))
    model.eval()


    text_tokens = ska_tokenizer.tokenize_raw_text(question)
    encoded_tokens = ska_tokenizer.encode_list(text_tokens)
    seq_len = skalm_config.seq_len
    encoded_tokens = encoded_tokens[-seq_len:]
    encoded_tokens_len = len(encoded_tokens)
    final_encoded_tokens: list[int] = []
    if encoded_tokens_len < seq_len:
        missing = seq_len - encoded_tokens_len
        final_encoded_tokens = [ska_tokenizer.encoded_token_pad for _ in range(missing)]

    final_encoded_tokens.extend(encoded_tokens)
    print('final_encoded_tokens', final_encoded_tokens)


    # X = X.reshape(n_patterns, seq_len)

    assert seq_len == len(final_encoded_tokens)


    total_tokens = 0
    total_eos = 0
    predicted_encoded_tokens: list[int] = []
    while total_tokens < skalm_config.max_tokens and total_eos < skalm_config.max_eos:
        X = torch.tensor([final_encoded_tokens], dtype=torch.int64)
        predicted_encoded_tensor = predict(model, X)
        predicted_encoded_token: int = int(predicted_encoded_tensor[0].item())
        predicted_encoded_tokens.append(predicted_encoded_token)
        print('predicted_encoded_token', predicted_encoded_token)
        final_encoded_tokens.pop(0)
        final_encoded_tokens.append(predicted_encoded_token)

        total_tokens += 1
        if predicted_encoded_token == ska_tokenizer.encoded_token_eos:
            total_eos += 1


    print('predicted_encoded_tokens', predicted_encoded_tokens)
    predicted_tokens = ska_tokenizer.decode_list(predicted_encoded_tokens)
    print('predicted_tokens', predicted_tokens)
    # filter unknown tokens
    predicted_tokens = list(filter(lambda x: x != ska_tokenizer.token_unk, predicted_tokens))
    # replace EOS with newline
    predicted_tokens = list(map(lambda x: x if x != ska_tokenizer.token_eos else "\n", predicted_tokens))
    answer = " ".join(predicted_tokens)
    print('answer', answer)
    return answer