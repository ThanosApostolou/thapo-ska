use std::process;

use crate::domain::nn_model::service_nn_model;
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;

pub fn do_create_skalm(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_create_skalm start");

    let skalm_data = service_nn_model::get_skalm_data();

    let python_main_path = my_paths::get_ska_llm_main_py(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let data_path = my_paths::get_models_data_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let skalm_dir_path = my_paths::get_models_dir(&global_state.env_config)
        .join(&skalm_data.model_path)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let skalm_config_path = my_paths::get_skalm_config_file(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let ska_tmp_dir = my_paths::get_ska_tmp_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();

    py_create_skalm(
        python_main_path,
        data_path,
        skalm_dir_path,
        skalm_config_path,
        ska_tmp_dir,
    )?;
    tracing::trace!("do_create_skalm end");
    Ok(())
}

fn py_create_skalm(
    python_main_path: String,
    data_path: String,
    skalm_dir_path: String,
    skalm_config_path: String,
    ska_tmp_dir: String,
) -> anyhow::Result<()> {
    process::Command::new("python3")
        .args([
            python_main_path,
            "create_skalm".to_string(),
            data_path,
            skalm_dir_path,
            skalm_config_path,
            ska_tmp_dir,
        ])
        .spawn()?
        .wait()?;
    Ok(())
}
