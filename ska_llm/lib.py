import sys
from ska_llm.scripts import download_llms, rag


def download_llm_huggingface(downloadDir: str, nn_models: list[tuple[str, str, str, str]]):
    download_llms.download_llm_huggingface(downloadDir, nn_models)


def rag_prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    print('data_path, vector_store_path, embedding_model_path', (data_path, vector_store_path, embedding_model_path))
    rag.prepare(data_path, vector_store_path, embedding_model_path)


def rag_invoke(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, question: str, model_type: str):
    return rag.invoke(vector_store_path, embedding_model_path, llm_model_path, prompt_template, question, model_type)


if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])