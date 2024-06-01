use crate::{
    domain::nn_model::{service_nn_model, NnModelData, NnModelType},
    modules::{error::ErrorResponse, global_state::GlobalState},
};

use super::{DtoAssistantOptions, DtoLlmData};

pub async fn do_fetch_assistant_options(
    _global_state: &GlobalState,
) -> Result<DtoAssistantOptions, ErrorResponse> {
    let mut models: Vec<NnModelData> = vec![NnModelData::get_skalm_data()];
    models.extend(
        service_nn_model::get_nn_models_list()
            .into_iter()
            .filter(|nn_model| NnModelType::ModelLlm == nn_model.model_type)
            .collect::<Vec<NnModelData>>(),
    );

    let dto_llms: Vec<DtoLlmData> = models
        .into_iter()
        .map(|llm| DtoLlmData {
            name: llm.name,
            default_prompt: llm.default_prompt,
        })
        .collect();

    Ok(DtoAssistantOptions { llms: dto_llms })
}
