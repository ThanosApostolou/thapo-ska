use std::fs;
use std::process;

use crate::domain::nn_model::service_nn_model;
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;

pub fn do_download(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_download start");
    let contents = fs::read_to_string(global_state.env_config.ska_llm_dir.clone() + "/lib.py")
        .expect("Should have been able to read the file");

    let download_dir = my_paths::get_models_download_dir(&global_state.env_config);
    let download_dir = download_dir.as_path();
    if download_dir.exists() {
        fs::remove_dir_all(download_dir)?;
    }
    fs::create_dir_all(download_dir)?;

    println!("{}", contents);

    let nn_models = service_nn_model::get_nn_models_list();
    let python_main_path = my_paths::get_ska_llm_main_py(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let models_download_dir = my_paths::get_models_download_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    for nn_model in nn_models {
        //     dd_parser("")
        // parser_download_llm.add_argument('downloadDir')
        // parser_download_llm.add_argument('repo_id', type=str)
        // parser_download_llm.add_argument('rel_path', type=str)
        // parser_download_llm.add_argument('revision', type=str)
        // parser_download_llm.add_argument('allow_patterns
        py_download_model(
            python_main_path.clone(),
            models_download_dir.clone(),
            nn_model.repo_id.clone(),
            nn_model.rel_path.clone(),
            nn_model.revision.clone(),
            nn_model.allow_patterns.clone(),
            nn_model.ignore_patterns.clone(),
        )?;
    }
    tracing::trace!("do_download end");
    Ok(())
}

fn py_download_model(
    python_main_path: String,
    download_dir: String,
    repo_id: String,
    rel_path: String,
    revision: String,
    allow_patterns: String,
    ignore_patterns: String,
) -> anyhow::Result<()> {
    process::Command::new("python3")
        .args([
            python_main_path,
            "download_llm".to_string(),
            download_dir,
            repo_id,
            rel_path,
            revision,
            allow_patterns,
            ignore_patterns,
        ])
        .spawn()?
        .wait()?;
    Ok(())
}
