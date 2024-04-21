use std::process;

use crate::domain::nn_model::{service_nn_model, NnModelType};
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;

pub fn do_rag_prepare(global_state: &GlobalState, emb_name: &String) -> anyhow::Result<()> {
    tracing::trace!("do_rag_prepare start");
    let nn_models = service_nn_model::get_nn_models_list();
    let emb_model_data = nn_models
        .iter()
        .filter(|nn_model| nn_model.name == *emb_name)
        .next()
        .ok_or(anyhow::anyhow!(
            "could not find nn_model with name {}",
            emb_name
        ))?;

    if !matches!(emb_model_data.model_type, NnModelType::ModelEmbedding) {
        return Err(anyhow::anyhow!(
            "{} is not an embedding model",
            emb_model_data.name
        ));
    }
    let python_lib_path = my_paths::get_ska_llm_lib_py(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let data_path = my_paths::get_models_data_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let vector_store_path = my_paths::get_vector_store_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let embedding_model_path = my_paths::get_models_dir(&global_state.env_config)
        .join(&emb_model_data.rel_path)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    py_rag_prepare(
        python_lib_path,
        data_path,
        vector_store_path,
        embedding_model_path,
    )?;
    tracing::trace!("do_rag_prepare end");
    Ok(())
}

fn py_rag_prepare(
    python_lib_path: String,
    data_path: String,
    vector_store_path: String,
    embedding_model_path: String,
) -> anyhow::Result<()> {
    process::Command::new("python3")
        .args([
            python_lib_path,
            "rag_prepare".to_string(),
            data_path,
            vector_store_path,
            embedding_model_path,
        ])
        .spawn()?
        .wait()?;
    Ok(())
}
