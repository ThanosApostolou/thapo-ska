use sea_orm::{entity::prelude::*, ColumnTrait, EntityTrait, QueryFilter};

use crate::domain::entities::users;

pub async fn find_by_sub(
    db_connection: &impl ConnectionTrait,
    sub: &str,
) -> anyhow::Result<Option<users::Model>> {
    tracing::trace!("repo_users::find_by_sub start {}", sub);
    let user = users::Entity::find()
        .filter(users::Column::Sub.eq(sub))
        .one(db_connection)
        .await?;
    tracing::trace!("repo_users::find_by_sub end {}", sub);
    Ok(user)
}

pub async fn insert(
    db_connection: &impl ConnectionTrait,
    user_am: users::ActiveModel,
) -> anyhow::Result<users::Model> {
    tracing::trace!("repo_users::insert start user.sub={:?}", &user_am.sub);
    let user = user_am.insert(db_connection).await?;
    tracing::trace!("repo_users::insert end user.sub={:?}", &user.sub);
    Ok(user)
}

pub async fn update(
    db_connection: &impl ConnectionTrait,
    user_am: users::ActiveModel,
) -> anyhow::Result<users::Model> {
    tracing::trace!("repo_users::insert start user.sub={:?}", &user_am.sub);
    let user = user_am.update(db_connection).await?;
    tracing::trace!("repo_users::insert end user.sub={:?}", &user.sub);
    Ok(user)
}
