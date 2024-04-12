use std::{fs, path::Path};

use pyo3::types::PyList;

use crate::domain::nn_model::{service_nn_model, NnModelData};
use crate::modules::db;
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;
use crate::prelude::*;

pub async fn do_migrate(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_migrate start");

    db::migrate_db(&global_state.db_connection).await?;
    tracing::trace!("do_migrate end");
    Ok(())
}
