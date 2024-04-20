import argparse
import os
import ska_llm.lib as skalib

HOME = os.environ['HOME']

def rag_prepare():
    data_path = f"{os.environ['HOME']}/.config/ska/local/data"
    vectore_store_path = f"{os.environ['HOME']}/.config/ska/local/vector_store"
    embedding_model_path = f"{os.environ['HOME']}/.config/ska/local/llms/all-MiniLM-L6-v2"
    print(f"rag {vectore_store_path}")
    skalib.rag_prepare(data_path, vectore_store_path, embedding_model_path)


def main():
    parser = argparse.ArgumentParser(
                    prog='ska_llm_main',
                    description='ska_llm_main',
                    epilog='ska_llm_main')
    parser.add_argument('action', action='store', choices=['rag_prepare'])
    args = parser.parse_args()

    if args.action == 'rag_prepare':
        rag_prepare()



if __name__ == "__main__":
    main()