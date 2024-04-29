import argparse
import sys
import skalib

def handle_download_llm_huggingface(downloadDir: str, repo_id: str, rel_path: str, revision: str, allow_patterns: str, ignore_patterns: str):
    skalib.download_llm_huggingface(downloadDir, repo_id, rel_path, revision, allow_patterns, ignore_patterns)


def handle_rag_prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    skalib.rag_prepare(data_path, vector_store_path, embedding_model_path)


def handle_rag_invoke(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, question: str, model_type: str):
    skalib.rag_invoke(vector_store_path, embedding_model_path, llm_model_path, prompt_template, question, model_type)


def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    subparsers = parser.add_subparsers(dest='cmd')
    parser_download_llm = subparsers.add_parser("download_llm")
    parser_download_llm.add_argument('downloadDir')
    parser_download_llm.add_argument('repo_id', type=str)
    parser_download_llm.add_argument('rel_path', type=str)
    parser_download_llm.add_argument('revision', type=str)
    parser_download_llm.add_argument('allow_patterns', type=str)
    parser_download_llm.add_argument('ignore_patterns', type=str)

    parser_rag_prepare = subparsers.add_parser("rag_prepare")
    parser_rag_prepare.add_argument('data_path', type=str)
    parser_rag_prepare.add_argument('vector_store_path', type=str)
    parser_rag_prepare.add_argument('embedding_model_path', type=str)

    parser_rag_invoke = subparsers.add_parser("rag_invoke")
    parser_rag_invoke.add_argument('vector_store_path', type=str)
    parser_rag_invoke.add_argument('embedding_model_path', type=str)
    parser_rag_invoke.add_argument('llm_model_path', type=str)
    parser_rag_invoke.add_argument('prompt_template', type=str)
    parser_rag_invoke.add_argument('question', type=str)
    parser_rag_invoke.add_argument('model_type', type=str)
    args = parser.parse_args()

    if args.cmd == 'download_llm':
        handle_download_llm_huggingface(args.downloadDir, args.repo_id, args.rel_path, args.revision, args.allow_patterns, args.ignore_patterns)
    elif args.cmd == 'rag_prepare':
        handle_rag_prepare(args.data_path, args.vector_store_path, args.embedding_model_path)
    elif args.cmd == 'rag_invoke':
        handle_rag_invoke(args.vector_store_path, args.embedding_model_path, args.llm_model_path, args.prompt_template, args.question, args.model_type)
    else:
        raise Exception(f"unknown cmd {args.cmd}")



if __name__ == '__main__':
    main()