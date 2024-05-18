import argparse
import os
import sys
import skalib

HOME = os.environ['HOME']
data_path = f"{os.environ['HOME']}/.config/ska/local/data"


def rag_prepare():
    vectore_store_path = f"{os.environ['HOME']}/.config/ska/local/vector_store"
    embedding_model_path = f"{os.environ['HOME']}/.config/ska/local/llms/all-MiniLM-L6-v2"
    print(f"rag {vectore_store_path}")
    skalib.rag_prepare(data_path, vectore_store_path, embedding_model_path)

def rag_invoke():
    vectore_store_path = f"{os.environ['HOME']}/.config/ska/local/vector_store"
    embedding_model_path = f"{os.environ['HOME']}/.config/ska/local/llms/all-MiniLM-L6-v2"
    llm_model_path = f"{os.environ['HOME']}/.config/ska/local/llms/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf"
    prompt_template="""
[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>>
Question: {question}
Context: {context}
Answer: [/INST]
"""
    model_type = "llamacpp"
    print(f"rag {vectore_store_path}")
    (context, question, answer) = skalib.rag_invoke(vectore_store_path, embedding_model_path, llm_model_path, prompt_template, "what is object oriented programming?", model_type)
    print("context")
    print(context)
    print("question")
    print(question)
    print("answer")
    print(answer)

def create_thapollm():
    skalib.create_thapollm(data_path)

def main():
    parser = argparse.ArgumentParser(
                    prog='ska_llm_main',
                    description='ska_llm_main',
                    epilog='ska_llm_main')
    parser.add_argument('action', action='store', choices=['rag_prepare', 'rag_invoke'])
    args = parser.parse_args()

    if args.action == 'rag_prepare':
        rag_prepare()
    elif args.action == 'rag_invoke':
        rag_invoke()
    elif args.action == 'create_thapollm':
        create_thapollm()



def hello(args: list[tuple[str, str, str]]):
    print(args)
    arg0 = args[0]
    print(arg0)


if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])