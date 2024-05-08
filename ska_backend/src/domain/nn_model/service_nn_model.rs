use std::{path::PathBuf, process};

use crate::{
    domain::nn_model::{InvokeOutputDto, NnModelType},
    modules::{
        global_state::{EnvConfig, GlobalState},
        myfs::my_paths,
    },
};

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

pub fn rag_invoke(
    global_state: &GlobalState,
    emb_name: &String,
    llm_name: &String,
    question: &String,
    prompt_template: &Option<String>,
) -> anyhow::Result<InvokeOutputDto> {
    tracing::trace!("action_rag_invoke start");
    // validate syntax
    if emb_name.len() > 96 {
        return Err(anyhow::anyhow!("emb_name length must be <= 96"));
    }
    if llm_name.len() > 96 {
        return Err(anyhow::anyhow!("llm_name length must be <= 96"));
    }
    if question.len() > 256 {
        return Err(anyhow::anyhow!("question length must be <= 256"));
    }
    if let Some(prompt_template) = prompt_template {
        if prompt_template.len() > 384 {
            return Err(anyhow::anyhow!("prompt_template length must be <= 384"));
        }
    }

    let nn_models = get_nn_models_list();
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
    let prompt_template = match prompt_template {
        Some(pt) => pt.clone(),
        None => llm_model_data.default_prompt.clone(),
    };
    let llm_model_type = match &llm_model_data.llm_model_type {
        Some(llm_model_type) => llm_model_type.get_value(),
        None => "",
    };
    let invoke_output_dto = py_rag_invoke(
        python_lib_path,
        vector_store_path,
        embedding_model_path,
        llm_model_path,
        prompt_template,
        question.clone(),
        llm_model_type.to_string(),
    )?;
    tracing::trace!("action_rag_invoke end");
    Ok(invoke_output_dto)
}

fn py_rag_invoke(
    python_lib_path: String,
    vector_store_path: String,
    embedding_model_path: String,
    llm_model_path: String,
    prompt_template: String,
    question: String,
    llm_model_type: String,
) -> anyhow::Result<InvokeOutputDto> {
    let output = process::Command::new("python3")
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
        .output()?;
    if !output.status.success() {
        let error_str = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(anyhow::anyhow!(error_str));
    }
    let output_str = String::from_utf8_lossy(&output.stdout).to_string();
    let rag_invoke_output: Vec<&str> = output_str.split("rag_invoke_output:").collect();
    if rag_invoke_output.len() != 2 {
        return Err(anyhow::anyhow!("rag_invoke_output should have length 2"));
    }
    let output_json = rag_invoke_output
        .last()
        .ok_or(anyhow::anyhow!("rag_invoke_output couldn't get last"))?
        .trim();

    let invoke_output_dto: InvokeOutputDto = serde_json::from_str(output_json)?;
    tracing::debug!("{:?}", invoke_output_dto);

    Ok(invoke_output_dto)
}
