use sea_orm_migration::prelude::*;
#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Users::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Users::UserId)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Users::Sub).string().not_null().unique_key())
                    .col(
                        ColumnDef::new(Users::Email)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(ColumnDef::new(Users::LastLogin).date_time().not_null())
                    .col(ColumnDef::new(Users::CreatedAt).date_time().not_null())
                    .col(ColumnDef::new(Users::UpdatedAt).date_time().not_null())
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(UserChatMessage::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(UserChatMessage::UserChatMessageId)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(
                        ColumnDef::new(UserChatMessage::UserIdFk)
                            .big_integer()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(UserChatMessage::MessageType)
                            .string()
                            .not_null(),
                    )
                    .col(ColumnDef::new(UserChatMessage::Message).string().not_null())
                    .col(
                        ColumnDef::new(UserChatMessage::CreatedAt)
                            .date_time()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(UserChatMessage::UpdatedAt)
                            .date_time()
                            .not_null(),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .from(UserChatMessage::Table, UserChatMessage::UserIdFk)
                            .to(Users::Table, Users::UserId),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(NnModel::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(NnModel::NnModelId)
                            .big_integer()
                            .not_null()
                            .auto_increment()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(NnModel::Path).string().not_null())
                    .col(ColumnDef::new(NnModel::IsTrained).boolean().not_null())
                    .col(ColumnDef::new(NnModel::CreatedAt).date_time().not_null())
                    .col(ColumnDef::new(NnModel::UpdatedAt).date_time().not_null())
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        if manager.has_table(Users::Table.to_string()).await? {
            manager
                .drop_table(Table::drop().cascade().table(Users::Table).to_owned())
                .await?;
        }
        if manager
            .has_table(UserChatMessage::Table.to_string())
            .await?
        {
            manager
                .drop_table(
                    Table::drop()
                        .cascade()
                        .table(UserChatMessage::Table)
                        .to_owned(),
                )
                .await?;
        }
        if manager.has_table(NnModel::Table.to_string()).await? {
            manager
                .drop_table(Table::drop().cascade().table(NnModel::Table).to_owned())
                .await?;
        }

        Ok(())
    }
}

#[derive(DeriveIden)]
pub enum Users {
    Table,
    UserId,
    Sub,
    Email,
    LastLogin,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
pub enum UserChatMessage {
    Table,
    UserChatMessageId,
    UserIdFk,
    MessageType,
    Message,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
pub enum NnModel {
    Table,
    NnModelId,
    Path,
    IsTrained,
    CreatedAt,
    UpdatedAt,
}
