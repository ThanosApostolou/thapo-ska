from huggingface_hub import snapshot_download
from ska_llm.scripts.hello import hello

def download_llm_huggingface():
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="./dist/llms/all-MiniLM-L6-v2")
    snapshot_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", allow_patterns="llama-2-7b-chat.ggmlv3.q8_0.bin", local_dir="./dist/llms/llama-2-7b-chat")


def main():
    hello()
    download_llm_huggingface()

if __name__ == "__main__":
    main()