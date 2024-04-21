from huggingface_hub import snapshot_download



def download_llm_huggingface(downloadDir: str, nn_models: list[tuple[str, str, str, str]]):
    print("nn_models:")
    print(nn_models)

    for nn_model in nn_models:
        (repo_id, rel_path, revision, allow_patterns) = nn_model
        snapshot_download(repo_id=repo_id, local_dir=downloadDir + "/" + rel_path, revision=revision, allow_patterns=allow_patterns, local_dir_use_symlinks=False)


    # snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=downloadDir + "/all-MiniLM-L6-v2")
    # snapshot_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", allow_patterns="llama-2-7b-chat.ggmlv3.q8_0.bin", local_dir=downloadDir + "/llama-2-7b-chat")


def main():
    download_llm_huggingface("./dist/llms", [])

if __name__ == "__main__":
    main()