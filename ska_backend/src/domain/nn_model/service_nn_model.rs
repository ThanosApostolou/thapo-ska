use std::path::PathBuf;

use crate::modules::global_state::EnvConfig;

use super::NnModelData;

pub fn get_models_download_dir(env_config: &EnvConfig) -> PathBuf {
    let download_dir_str = env_config.ska_tmp_dir.clone() + "/llms";
    let download_dir = PathBuf::from(&download_dir_str);
    download_dir
}

pub fn get_models_dir(env_config: &EnvConfig) -> PathBuf {
    let models_dir_str = env_config.ska_user_conf_dir.clone() + "/files/llms";
    let models_dir = PathBuf::from(&models_dir_str);
    models_dir
}

pub fn get_nn_models_list() -> Vec<NnModelData> {
    let mut nn_models: Vec<NnModelData> = vec![];

    nn_models.push(NnModelData {
        repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        rel_path: "all-MiniLM-L6-v2".to_string(),
        revision: "e4ce9877abf3edfe10b0d82785e83bdcb973e22e".to_string(),
        allow_patterns: "*".to_string(),
    });

    nn_models.push(NnModelData {
        repo_id: "TheBloke/Llama-2-7B-Chat-GGUF".to_string(),
        rel_path: "Llama-2-7B-Chat-GGUF".to_string(),
        revision: "191239b3e26b2882fb562ffccdd1cf0f65402adb".to_string(),
        allow_patterns: "llama-2-7b-chat.Q2_K.gguf".to_string(),
    });

    nn_models.push(NnModelData {
        repo_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        rel_path: "TinyLlama-1.1B-Chat".to_string(),
        revision: "fe8a4ea1ffedaf415f4da2f062534de366a451e6".to_string(),
        allow_patterns: "*".to_string(),
    });

    // nn_models.push(NnModelData {
    //     repo_id: "google/gemma-2b-it".to_string(),
    //     rel_path: "gemma-2b-it".to_string(),
    //     revision: "060189a16d5d2713425599b533a9e8ece8f5cca6".to_string(),
    //     allow_patterns: "*".to_string(),
    // });
    nn_models
}
