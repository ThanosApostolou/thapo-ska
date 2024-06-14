//! `SeaORM` Entity. Generated by sea-orm-codegen 0.12.15

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "thapo_ska_schema", table_name = "user_chat")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub chat_id: i64,
    pub user_id_fk: i64,
    pub chat_name: String,
    pub llm_model: String,
    pub prompt: Option<String>,
    #[sea_orm(column_type = "Double", nullable)]
    pub temperature: Option<f64>,
    #[sea_orm(column_type = "Double", nullable)]
    pub top_p: Option<f64>,
    pub created_at: DateTime,
    pub updated_at: DateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::chat_message::Entity")]
    ChatMessage,
    #[sea_orm(
        belongs_to = "super::users::Entity",
        from = "Column::UserIdFk",
        to = "super::users::Column::UserId",
        on_update = "NoAction",
        on_delete = "NoAction"
    )]
    Users,
}

impl Related<super::chat_message::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ChatMessage.def()
    }
}

impl Related<super::users::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Users.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
