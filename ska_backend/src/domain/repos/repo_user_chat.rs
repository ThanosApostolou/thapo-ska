use sea_orm::{
    entity::prelude::*, ColumnTrait, DeleteResult, EntityTrait, Order, QueryFilter, QueryOrder,
};

use crate::domain::entities::{chat_message, user_chat};

pub async fn find_by_user_id(
    db_connection: &impl ConnectionTrait,
    user_id: i64,
) -> anyhow::Result<Vec<user_chat::Model>> {
    tracing::trace!("select_by_user_id start user_id={}", user_id);
    let user_chats = user_chat::Entity::find()
        .filter(user_chat::Column::UserIdFk.eq(user_id))
        .order_by(user_chat::Column::CreatedAt, Order::Asc)
        .all(db_connection)
        .await?;
    tracing::trace!("select_by_user_id end user_id={}", user_id);
    Ok(user_chats)
}

pub async fn find_by_chat_id(
    db_connection: &impl ConnectionTrait,
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

pub async fn find_by_chat_id_with_messages(
    db_connection: &impl ConnectionTrait,
    chat_id: i64,
) -> anyhow::Result<Option<(user_chat::Model, Vec<chat_message::Model>)>> {
    tracing::trace!("find_by_chat_id_with_messages start chat_id={}", chat_id);
    let user_chats = user_chat::Entity::find()
        .find_with_related(chat_message::Entity)
        .filter(user_chat::Column::ChatId.eq(chat_id))
        .order_by(user_chat::Column::ChatId, Order::Asc)
        .order_by(chat_message::Column::CreatedAt, Order::Asc)
        .all(db_connection)
        .await?;
    if user_chats.len() > 1 {
        return Err(anyhow::anyhow!(format!(
            "user_chats has len {}",
            user_chats.len()
        )));
    }
    let user_chat = user_chats.get(0).cloned();
    tracing::trace!("find_by_chat_id_with_messages end chat_id={}", chat_id);
    return Ok(user_chat);
}

pub async fn insert(
    db_connection: &impl ConnectionTrait,
    user_chat_am: user_chat::ActiveModel,
) -> anyhow::Result<user_chat::Model> {
    tracing::trace!("insert start chat_id={:?}", &user_chat_am.chat_id);
    let user_chat = user_chat_am.insert(db_connection).await?;
    tracing::trace!("insert end chat_id={:?}", &user_chat.chat_id);
    Ok(user_chat)
}

pub async fn update(
    db_connection: &impl ConnectionTrait,
    user_chat_am: user_chat::ActiveModel,
) -> anyhow::Result<user_chat::Model> {
    tracing::trace!("update start chat_id={:?}", &user_chat_am.chat_id);
    let user_chat = user_chat_am.update(db_connection).await?;
    tracing::trace!("update end chat_id={:?}", &user_chat.chat_id);
    Ok(user_chat)
}

pub async fn delete(
    db_connection: &impl ConnectionTrait,
    user_chat_am: user_chat::ActiveModel,
) -> anyhow::Result<DeleteResult> {
    tracing::trace!("delete start chat_id={:?}", &user_chat_am.chat_id);
    let delete_result: DeleteResult = user_chat_am.delete(db_connection).await?;
    tracing::trace!(
        "delete end rows_affected={:?}",
        &delete_result.rows_affected
    );
    Ok(delete_result)
}
