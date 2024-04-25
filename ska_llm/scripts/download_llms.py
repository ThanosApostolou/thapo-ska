from huggingface_hub import snapshot_download



def download_llm_huggingface(downloadDir: str, repo_id: str, rel_path: str, revision: str, allow_patterns: str, ignore_patterns: str):
    print("nn_models:")
    allow_patterns_list = allow_patterns.split(",")
    ignore_patterns_list = ignore_patterns.split(",")
    # print(nn_models)

    # for nn_model in nn_models:
    #     (repo_id, rel_path, revision, allow_patterns) = nn_model
    snapshot_download(repo_id=repo_id, local_dir=downloadDir + "/" + rel_path, revision=revision, allow_patterns=allow_patterns_list, ignore_patterns=ignore_patterns_list, local_dir_use_symlinks=False)


    # snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=downloadDir + "/all-MiniLM-L6-v2")
    # snapshot_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", allow_patterns="llama-2-7b-chat.ggmlv3.q8_0.bin", local_dir=downloadDir + "/llama-2-7b-chat")
