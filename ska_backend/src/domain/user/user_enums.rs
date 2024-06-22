use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    AsRefStr,
    IntoStaticStr,
    EnumString,
    Display,
    PartialEq,
    Eq,
    Hash,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum ChatPacketType {
    #[serde(rename = "QUESTION")]
    Question,
    #[serde(rename = "ANSWER")]
    Answer,
}
