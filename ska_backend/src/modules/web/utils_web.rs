use serde::{Deserialize, Serialize};

use crate::modules::error::ErrorResponse;

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum ControllerResponse<T>
where
    T: serde::Serialize,
{
    Ok(T),
    Err(ErrorResponse),
}
