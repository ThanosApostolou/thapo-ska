from ska_llm.scripts import download_llms, rag


def download_llm_huggingface(downloadDir: str, nn_models: list[tuple[str, str, str, str]]):
    download_llms.download_llm_huggingface(downloadDir, nn_models)


def rag_prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    rag.prepare(data_path, vector_store_path, embedding_model_path)