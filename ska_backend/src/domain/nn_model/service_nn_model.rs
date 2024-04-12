use std::path::PathBuf;

use crate::modules::global_state::EnvConfig;

use super::{NnModelData, NnModelEnum};


pub fn get_models_dir(env_config: &EnvConfig) -> PathBuf {
    let models_dir_str = env_config.ska_user_conf_dir.clone() + "/files/llms";
    let models_dir = PathBuf::from(&models_dir_str);
    models_dir
}

pub fn get_nn_models_list() -> Vec<NnModelData> {

    // nn_models.push(NnModelData {
    //     repo_id: "google/gemma-2b-it".to_string(),
    //     rel_path: "gemma-2b-it".to_string(),
    //     revision: "060189a16d5d2713425599b533a9e8ece8f5cca6".to_string(),
    //     allow_patterns: "*".to_string(),
    // });
    NnModelEnum::get_data_list()
}
