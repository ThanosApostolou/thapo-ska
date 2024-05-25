use std::path::PathBuf;

use crate::modules::global_state::EnvConfig;

// Paths defined in EnvConfig

pub fn get_ska_conf_dir(env_config: &EnvConfig) -> PathBuf {
    PathBuf::from(&env_config.ska_conf_dir)
}

pub fn get_ska_data_dir(env_config: &EnvConfig) -> PathBuf {
    PathBuf::from(&env_config.ska_data_dir)
}

pub fn get_ska_tmp_dir(env_config: &EnvConfig) -> PathBuf {
    PathBuf::from(&env_config.ska_tmp_dir)
}

pub fn get_ska_user_conf_dir(env_config: &EnvConfig) -> PathBuf {
    PathBuf::from(&env_config.ska_user_conf_dir)
}

pub fn get_ska_llm_dir(env_config: &EnvConfig) -> PathBuf {
    PathBuf::from(&env_config.ska_llm_dir)
}

pub fn get_ska_llm_main_py(env_config: &EnvConfig) -> PathBuf {
    get_ska_llm_dir(env_config).join("main.py")
}

// Derived paths used in app

pub fn get_models_download_dir(env_config: &EnvConfig) -> PathBuf {
    get_ska_tmp_dir(env_config).join("llms")
}

pub fn get_models_dir(env_config: &EnvConfig) -> PathBuf {
    get_ska_user_conf_dir(env_config).join("llms")
}

pub fn get_models_download_data_dir(env_config: &EnvConfig) -> PathBuf {
    get_ska_tmp_dir(env_config).join("data")
}

pub fn get_models_data_dir(env_config: &EnvConfig) -> PathBuf {
    get_ska_user_conf_dir(env_config).join("data")
}

pub fn get_vector_store_dir(env_config: &EnvConfig) -> PathBuf {
    get_ska_user_conf_dir(env_config).join("vector_store")
}

pub fn get_skalm_config_file(env_config: &EnvConfig) -> PathBuf {
    get_ska_conf_dir(env_config).join("skalm_config.json")
}
