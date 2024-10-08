import math
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
import pickle
import os
import random
import torch

from scripts import rag
from scripts import constants
from scripts.skalm.skalm import Skalm, predict, train_skallm_lstm
from scripts.skalm.skalm_config import SkalmConfig, SkalmProps
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
            raw_text += " \n " + doc.page_content.strip().lower()
            # doc_chars = set(doc.page_content.lower())
            # vocab_set = vocab_set.union(doc_chars)

        with open(raw_text_pickle, 'wb') as f:
            pickle.dump(raw_text, f)

    return raw_text


def create_data_X_Y(ska_tokenizer: SkaTokenizer, skalm_config: SkalmConfig, sentences: list[str]) -> tuple[list[list[int]], list[int]]:
    seq_len = skalm_config.skalm_props.seq_len
    tokens: list[str] = ska_tokenizer.tokenize_sentences(sentences, skalm_config)
    encoded_tokens: list[int] = ska_tokenizer.encode_list(tokens)
    encoded_tokens_len = len(encoded_tokens)

    dataX: list[list[int]] = []
    dataY: list[int] = []
    for i in range(0, encoded_tokens_len - seq_len, 1):
        seq_in = encoded_tokens[i:i + seq_len]
        seq_out = encoded_tokens[i + seq_len]
        dataX.append(seq_in)
        dataY.append(seq_out)


    assert len(seq_in) == seq_len

    return dataX, dataY


def create_skalm(data_path: str, skalm_dir_path: str, skalm_config_path: str, ska_tmp_dir: str):
    print("create_thapollm start")
    if not os.path.exists(ska_tmp_dir):
        os.makedirs(ska_tmp_dir, exist_ok=True)

    skalm_config = SkalmConfig.from_json_file(skalm_config_path)
    raw_text = read_docs(data_path, ska_tmp_dir)

    # text_tokens = tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_WORD)

    ska_tokenizer = SkaTokenizer.from_raw_text(raw_text, skalm_config)
    seq_len = skalm_config.skalm_props.seq_len
    sentences: list[str] = ska_tokenizer.tokenize_text(raw_text, constants.TOKENIZE_METHOD_NLTK_SENT)
    # we use all the sentences for train data, since we don't want to lose any information
    dataXtrain, dataYtrain = create_data_X_Y(ska_tokenizer, skalm_config, sentences)

    # we use 20% of sentences for test data.
    # Test data exists in train data as well but we don't mind overtraining much because
    # we want skalm to answer only knowledge found in provided documents and not be a general purpose LLM
    K: int = math.floor(0.2 * len(sentences))
    test_indices = random.sample(range(len(sentences)), K)
    test_stences: list[str] = [sentences[i] for i in sorted(test_indices)]
    dataXtest, dataYtest = create_data_X_Y(ska_tokenizer, skalm_config, test_stences)


    print('Xtrain', list(map(lambda row: row[:20], dataXtrain[:20])))
    print('Xtrain', dataYtrain[:20])

    n_patterns_train = len(dataXtrain)
    Xtrain = torch.tensor(dataXtrain, dtype=torch.int64)
    print('X.shape', Xtrain.shape)
    Xtrain = Xtrain.reshape(n_patterns_train, seq_len)
    print('X.shape', Xtrain.shape)
    # X = X / float(ska_tokenizer.vocab_len)
    Ytrain = torch.tensor(dataYtrain)
    print(Xtrain.shape, Ytrain.shape)

    n_patterns_test = len(dataXtest)
    Xtest = torch.tensor(dataXtest, dtype=torch.int64)
    print('Xtest.shape', Xtest.shape)
    Xtest = Xtest.reshape(n_patterns_test, seq_len)
    print('Xtest.shape', Xtest.shape)
    Ytest = torch.tensor(dataYtest)
    print(Xtest.shape, Ytest.shape)

    skalm_config.skalm_props.save_json_file(skalm_dir_path)
    ska_tokenizer.save_json_file(skalm_dir_path)
    model = Skalm(skalm_config.skalm_props, n_vocab=ska_tokenizer.vocab_len)
    best_model_state_dict, train_losses, test_losses = train_skallm_lstm(model, skalm_config, skalm_dir_path, Xtrain, Ytrain, Xtest, Ytest)



def invoke_skalm(question: str, skalm_dir_path: str, skalm_config_path: str) -> rag.InvokeOutput:
    skalm_config = SkalmConfig.from_json_file(skalm_config_path)
    skalm_props = SkalmProps.from_json_file(f"{skalm_dir_path}/skalm_props.json")
    ska_tokenizer = SkaTokenizer.from_json_file(skalm_dir_path)
    model = Skalm(skalm_props, n_vocab=ska_tokenizer.vocab_len)
    # TODO use skalm.pth
    model.load_state_dict(torch.load(f"{skalm_dir_path}/skalm.pth"))
    model.eval()

    sentences: list[str] = ska_tokenizer.tokenize_text(question, constants.TOKENIZE_METHOD_NLTK_SENT)
    text_tokens = ska_tokenizer.tokenize_sentences(sentences, skalm_config)
    encoded_tokens = ska_tokenizer.encode_list(text_tokens)
    seq_len = skalm_props.seq_len
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
    invoke_output = rag.InvokeOutput()
    invoke_output.question = question
    invoke_output.answer = answer
    return invoke_output