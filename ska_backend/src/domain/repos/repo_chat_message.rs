use sea_orm::{
    entity::prelude::*, ColumnTrait, DeleteResult, EntityTrait, InsertResult, QueryFilter,
};

use crate::domain::entities::{chat_message, user_chat};

// pub async fn find_by_user_id(
//     db_connection: &impl ConnectionTrait,
//     user_id: i64,
// ) -> anyhow::Result<Vec<chat_message::Model>> {
//     tracing::trace!("select_by_user_id start user_id={}", user_id);
//     let chat_messages = chat_message::Entity::find().find_also_related(chat_message::Relation::UserChat)
//         .filter(chat_message::Relation::UserChat::Column::UserIdFk.eq(user_id))
//         .order_by(chat_message::Column::CreatedAt, Order::Asc)
//         .all(db_connection)
//         .await?;
//     tracing::trace!("select_by_user_id end user_id={}", user_id);
//     Ok(chat_messages)
// }

pub async fn find_by_chat_id(
    db_connection: &impl ConnectionTrait,
    chat_message_id: i64,
) -> anyhow::Result<Option<chat_message::Model>> {
    tracing::trace!("find_by_chat_id start chat_message_id={}", chat_message_id);
    let chat_message = chat_message::Entity::find()
        .filter(chat_message::Column::ChatMessageId.eq(chat_message_id))
        .one(db_connection)
        .await?;
    tracing::trace!("find_by_chat_id end chat_message_id={}", chat_message_id);
    Ok(chat_message)
}

pub async fn insert(
    db_connection: &impl ConnectionTrait,
    chat_message_am: chat_message::ActiveModel,
) -> anyhow::Result<chat_message::Model> {
    tracing::trace!("insert start chat_id_fk={:?}", &chat_message_am.chat_id_fk);
    let chat_message = chat_message_am.insert(db_connection).await?;
    tracing::trace!("insert end chat_id_fk={:?}", &chat_message.chat_id_fk);
    Ok(chat_message)
}

pub async fn insert_many(
    db_connection: &impl ConnectionTrait,
    chat_message_ams: Vec<chat_message::ActiveModel>,
) -> anyhow::Result<InsertResult<chat_message::ActiveModel>> {
    tracing::trace!(
        "insert start chat_message_ams.len()={:?}",
        &chat_message_ams.len()
    );
    let insert_result = chat_message::Entity::insert_many(chat_message_ams)
        .exec(db_connection)
        .await?;
    tracing::trace!(
        "insert end last_insert_id={:?}",
        &insert_result.last_insert_id
    );
    Ok(insert_result)
}

pub async fn update(
    db_connection: &impl ConnectionTrait,
    chat_message_am: chat_message::ActiveModel,
) -> anyhow::Result<chat_message::Model> {
    tracing::trace!(
        "update start chat_id_fk={:?}, chat_message_id={:?}",
        &chat_message_am.chat_id_fk,
        &chat_message_am.chat_message_id
    );
    let chat_message = chat_message_am.update(db_connection).await?;
    tracing::trace!(
        "update end chat_id_fk={:?}, chat_message_id={:?}",
        &chat_message.chat_id_fk,
        &chat_message.chat_message_id
    );
    Ok(chat_message)
}

pub async fn delete(
    db_connection: &impl ConnectionTrait,
    chat_message_am: chat_message::ActiveModel,
) -> anyhow::Result<DeleteResult> {
    tracing::trace!(
        "delete start chat_id_fk={:?}, chat_message_id={:?}",
        &chat_message_am.chat_id_fk,
        &chat_message_am.chat_message_id
    );
    let delete_result: DeleteResult = chat_message_am.delete(db_connection).await?;
    tracing::trace!(
        "delete end rows_affected={:?}",
        &delete_result.rows_affected
    );
    Ok(delete_result)
}

pub async fn delete_all_of_chat(
    db_connection: &impl ConnectionTrait,
    chat_id: i64,
) -> anyhow::Result<DeleteResult> {
    tracing::trace!("delete start chat_id={:?}", chat_id);
    let delete_result: DeleteResult = chat_message::Entity::delete_many()
        .filter(chat_message::Column::ChatIdFk.eq(chat_id))
        .exec(db_connection)
        .await?;
    tracing::trace!(
        "delete end rows_affected={:?}",
        &delete_result.rows_affected
    );
    Ok(delete_result)
}
