use crate::domain::nn_model::service_nn_model::get_nn_models_list;
use crate::domain::nn_model::{service_nn_model, InvokeOutputDto, NnModelData, NnModelType};
use crate::modules::global_state::GlobalState;

pub fn action_rag_invoke(
    global_state: &GlobalState,
    emb_name: &String,
    llm_name: &String,
    question: &String,
    prompt_template: &Option<String>,
) -> anyhow::Result<InvokeOutputDto> {
    tracing::trace!("action_rag_invoke start");

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

    let llm_model_data = if NnModelData::get_skalm_data().name.eq(llm_name) {
        &NnModelData::get_skalm_data()
    } else {
        nn_models
            .iter()
            .filter(|nn_model| nn_model.name == *llm_name)
            .next()
            .ok_or(anyhow::anyhow!(
                "could not find nn_model with name {}",
                llm_name
            ))?
    };

    if !matches!(llm_model_data.model_type, NnModelType::ModelLlm) {
        return Err(anyhow::anyhow!(
            "{} is not an llm model",
            llm_model_data.name
        ));
    }

    let invoke_output_dto = service_nn_model::rag_invoke(
        global_state,
        emb_model_data,
        llm_model_data,
        question,
        prompt_template,
        None,
        None,
    )?;
    tracing::info!("{:?}", invoke_output_dto);

    tracing::trace!("action_rag_invoke end");
    Ok(invoke_output_dto)
}
