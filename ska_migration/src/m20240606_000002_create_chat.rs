use std::fs;

use sea_orm::Statement;
use sea_orm_migration::prelude::*;

use crate::utils_migration;
#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        for entry in utils_migration::get_up_files("m20240606_000002").unwrap() {
            let contents =
                fs::read_to_string(entry.path()).expect("Should have been able to read the file");

            println!("executing:\n {}", contents);
            db.execute_unprepared(&contents).await?;
        }

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        for entry in utils_migration::get_down_files("m20240606_000002").unwrap() {
            let contents =
                fs::read_to_string(entry.path()).expect("Should have been able to read the file");

            println!("executing:\n {}", contents);
            db.execute_unprepared(&contents).await?;
        }

        Ok(())
    }
}
