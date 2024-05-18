from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from scripts import rag
from scripts import constants

def read_docs(data_path: str):
    docs: list[Document] = rag.read_docs(data_path)

    raw_text: str = ""
    vocab_set: set[str] = set()

    for doc in docs:
        raw_text = raw_text + doc.page_content
        doc_chars = set(doc.page_content.lower())
        vocab_set = vocab_set.union(doc_chars)

    raw_text_len = len(raw_text)
    vocab = sorted(list(vocab_set))
    vocab_len = len(vocab)

    return raw_text, raw_text_len, vocab, vocab_len


def create_thapollm(data_path: str):
    print("create_thapollm start")
    (raw_text, raw_text_len, vocab, vocab_len) = read_docs(data_path)
    print('raw_text[:100]', raw_text[:100])
    print('raw_text_len', raw_text_len)
    print('vocab', vocab)
    print('vocab_len', vocab_len)
