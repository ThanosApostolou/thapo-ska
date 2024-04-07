from ska_llm.scripts import download_llms


def download_llm_huggingface(downloadDir: str, nn_models: list[tuple[str, str, str, str]]):
    download_llms.download_llm_huggingface(downloadDir, nn_models)


def main():
    print("main start")


if __name__ == "__main__":
    main()