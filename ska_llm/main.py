import argparse
import sys
import skalib

def handle_download_llm_huggingface(downloadDir: str, repo_id: str, rel_path: str, revision: str, allow_patterns: str, ignore_patterns: str):
    skalib.download_llm_huggingface(downloadDir, repo_id, rel_path, revision, allow_patterns, ignore_patterns)


def handle_rag_prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    skalib.rag_prepare(data_path, vector_store_path, embedding_model_path)


def handle_rag_invoke(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, question: str, model_type: str, temperature: int, top_p: int):
    skalib.rag_invoke(vector_store_path, embedding_model_path, llm_model_path, prompt_template, question, model_type, temperature, top_p)

def handle_create_skalm(data_path: str, skalm_dir_path: str, skalm_config_path: str, ska_tmp_dir: str):
    skalib.create_skalm(data_path, skalm_dir_path, skalm_config_path, ska_tmp_dir)


def handle_invoke_skalm(question: str, skalm_dir_path: str, skalm_config_path: str):
    skalib.invoke_skalm(question, skalm_dir_path, skalm_config_path)

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
    parser_rag_invoke.add_argument('temperature', type=float)
    parser_rag_invoke.add_argument('top_p', type=float)

    parser_create_thapollm = subparsers.add_parser("create_skalm")
    parser_create_thapollm.add_argument('data_path', type=str)
    parser_create_thapollm.add_argument('skalm_dir_path', type=str)
    parser_create_thapollm.add_argument('skalm_config_path', type=str)
    parser_create_thapollm.add_argument('ska_tmp_dir', type=str)

    parser_invoke_skalm = subparsers.add_parser("invoke_skalm")
    parser_invoke_skalm.add_argument('question', type=str)
    parser_invoke_skalm.add_argument('skalm_dir_path', type=str)
    parser_invoke_skalm.add_argument('skalm_config_path', type=str)

    args = parser.parse_args()

    if args.cmd == 'download_llm':
        handle_download_llm_huggingface(args.downloadDir, args.repo_id, args.rel_path, args.revision, args.allow_patterns, args.ignore_patterns)
    elif args.cmd == 'rag_prepare':
        handle_rag_prepare(args.data_path, args.vector_store_path, args.embedding_model_path)
    elif args.cmd == 'rag_invoke':
        handle_rag_invoke(args.vector_store_path, args.embedding_model_path, args.llm_model_path, args.prompt_template, args.question, args.model_type, args.temperature, args.top_p)
    elif args.cmd == 'create_skalm':
        handle_create_skalm(args.data_path, args.skalm_dir_path, args.skalm_config_path, args.ska_tmp_dir)
    elif args.cmd == 'invoke_skalm':
        handle_invoke_skalm(args.question, args.skalm_dir_path, args.skalm_config_path)
    else:
        raise Exception(f"unknown cmd {args.cmd}")



if __name__ == '__main__':
    main()