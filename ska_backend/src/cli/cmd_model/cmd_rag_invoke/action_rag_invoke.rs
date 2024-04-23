use std::process;

use crate::domain::nn_model::{service_nn_model, NnModelType};
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;

pub fn action_rag_invoke(
    global_state: &GlobalState,
    emb_name: &String,
    llm_name: &String,
    question: &String,
) -> anyhow::Result<()> {
    tracing::trace!("action_rag_invoke start");
    let nn_models = service_nn_model::get_nn_models_list();
    // embedding
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
    // llm
    let llm_model_data = nn_models
        .iter()
        .filter(|nn_model| nn_model.name == *llm_name)
        .next()
        .ok_or(anyhow::anyhow!(
            "could not find nn_model with name {}",
            llm_name
        ))?;

    if !matches!(llm_model_data.model_type, NnModelType::ModelLlm) {
        return Err(anyhow::anyhow!(
            "{} is not an llm model",
            llm_model_data.name
        ));
    }

    // TODO
    //  llm_model_path: str, prompt_template: str, question: str, model_type: str

    let python_lib_path = my_paths::get_ska_llm_main_py(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let vector_store_path = my_paths::get_vector_store_dir(&global_state.env_config)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let embedding_model_path = my_paths::get_models_dir(&global_state.env_config)
        .join(&emb_model_data.model_path)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let llm_model_path = my_paths::get_models_dir(&global_state.env_config)
        .join(&llm_model_data.model_path)
        .to_str()
        .ok_or(anyhow::anyhow!("path not string"))?
        .to_string();
    let prompt_template = llm_model_data.default_prompt.clone();
    let llm_model_type = match &llm_model_data.llm_model_type {
        Some(llm_model_type) => llm_model_type.get_value(),
        None => "",
    };
    py_rag_invoke(
        python_lib_path,
        vector_store_path,
        embedding_model_path,
        llm_model_path,
        prompt_template,
        question.clone(),
        llm_model_type.to_string(),
    )?;
    tracing::trace!("action_rag_invoke end");
    Ok(())
}

fn py_rag_invoke(
    python_lib_path: String,
    vector_store_path: String,
    embedding_model_path: String,
    llm_model_path: String,
    prompt_template: String,
    question: String,
    llm_model_type: String,
) -> anyhow::Result<()> {
    process::Command::new("python3")
        .args([
            python_lib_path,
            "rag_invoke".to_string(),
            vector_store_path,
            embedding_model_path,
            llm_model_path,
            prompt_template,
            question,
            llm_model_type,
        ])
        .spawn()?
        .wait()?;
    Ok(())
}
