use std::fs;

use anyhow::anyhow;

use crate::domain::nn_model::service_nn_model;
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::{my_paths, utils_fs};

pub fn do_insert(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_insert start");

    let download_dir = my_paths::get_models_download_dir(&global_state.env_config);
    let download_dir = download_dir.as_path();
    if !download_dir.exists() {
        return Err(anyhow!("download_dir not exists"));
    }

    let models_dir = my_paths::get_models_dir(&global_state.env_config);
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }

    let nn_models = service_nn_model::get_nn_models_list();
    for nn_model in nn_models {
        let model_tmp_dir = download_dir.join(&nn_model.rel_path);
        let model_dir = models_dir.join(&nn_model.rel_path);
        if model_tmp_dir.exists() {
            if !model_tmp_dir.is_dir() {
                return Err(anyhow!(
                    "model_tmp_dir {} is not dir",
                    model_tmp_dir.to_str().unwrap_or("")
                ));
            }
            utils_fs::copy_dir_all(model_tmp_dir, model_dir)?;
        }
    }

    tracing::trace!("do_insert end");
    Ok(())
}

