import sys
from ska_llm.scripts import download_llms, rag
import json

def download_llm_huggingface(downloadDir: str, repo_id: str, rel_path: str, revision: str, allow_patterns: str, ignore_patterns: str):
    download_llms.download_llm_huggingface(downloadDir, repo_id, rel_path, revision, allow_patterns, ignore_patterns)


def rag_prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    print('data_path, vector_store_path, embedding_model_path', (data_path, vector_store_path, embedding_model_path))
    rag.prepare(data_path, vector_store_path, embedding_model_path)


def rag_invoke(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, question: str, model_type: str):
    output = rag.invoke(vector_store_path, embedding_model_path, llm_model_path, prompt_template, question, model_type)
    # print('rag_invoke_context:', output.context)
    # print('rag_invoke_question:', output.question)
    # print('rag_invoke_answer:', output.answer)
    output_dto = rag.InvokeOutputDto.from_invoke_output(output)
    output_json = json.dumps(output_dto.to_json_obj())
    print("rag_invoke_output:\n", output_json)
    return output_json


if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])