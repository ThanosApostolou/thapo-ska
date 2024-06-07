use sea_orm::{entity::prelude::*, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};

use crate::domain::entities::user_chat;

pub async fn find_by_user_id(
    db_connection: &DatabaseConnection,
    user_id: i64,
) -> anyhow::Result<Vec<user_chat::Model>> {
    tracing::trace!("select_by_user_id start user_id={}", user_id);
    let user_chats = user_chat::Entity::find()
        .filter(user_chat::Column::UserIdFk.eq(user_id))
        .all(db_connection)
        .await?;
    tracing::trace!("select_by_user_id end user_id={}", user_id);
    Ok(user_chats)
}

pub async fn find_by_chat_id(
    db_connection: &DatabaseConnection,
    chat_id: i64,
) -> anyhow::Result<Option<user_chat::Model>> {
    tracing::trace!("find_by_chat_id start chat_id={}", chat_id);
    let user_chat = user_chat::Entity::find()
        .filter(user_chat::Column::ChatId.eq(chat_id))
        .one(db_connection)
        .await?;
    tracing::trace!("find_by_chat_id end chat_id={}", chat_id);
    Ok(user_chat)
}

pub async fn insert(
    db_connection: &DatabaseConnection,
    user_chat_am: user_chat::ActiveModel,
) -> anyhow::Result<user_chat::Model> {
    tracing::trace!("insert start chat_id={:?}", &user_chat_am.chat_id);
    let user_chat = user_chat_am.insert(db_connection).await?;
    tracing::trace!("insert end chat_id={:?}", &user_chat.chat_id);
    Ok(user_chat)
}

pub async fn update(
    db_connection: &DatabaseConnection,
    user_chat_am: user_chat::ActiveModel,
) -> anyhow::Result<user_chat::Model> {
    tracing::trace!("update start chat_id={:?}", &user_chat_am.chat_id);
    let user_chat = user_chat_am.update(db_connection).await?;
    tracing::trace!("update end chat_id={:?}", &user_chat.chat_id);
    Ok(user_chat)
}
