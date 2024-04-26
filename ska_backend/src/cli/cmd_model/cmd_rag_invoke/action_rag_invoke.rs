use crate::domain::nn_model::{service_nn_model, InvokeOutputDto};
use crate::modules::global_state::GlobalState;

pub fn action_rag_invoke(
    global_state: &GlobalState,
    emb_name: &String,
    llm_name: &String,
    question: &String,
    prompt_template: &Option<String>,
) -> anyhow::Result<InvokeOutputDto> {
    tracing::trace!("action_rag_invoke start");

    let invoke_output_dto =
        service_nn_model::rag_invoke(global_state, emb_name, llm_name, question, prompt_template)?;
    tracing::info!("{:?}", invoke_output_dto);

    tracing::trace!("action_rag_invoke end");
    Ok(invoke_output_dto)
}
