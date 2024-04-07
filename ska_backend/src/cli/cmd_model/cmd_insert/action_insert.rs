use std::{fs, path::Path};

use crate::modules::global_state::GlobalState;
use crate::prelude::*;

pub fn do_insert(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("action_download.do_download start");
    let ska_llm_lib_path_str = global_state.env_config.ska_data_dir.clone()
        + "/thapo_ska_py/ska_llm/scripts/download_llms.py";
    tracing::info!("path_str: {}", ska_llm_lib_path_str);
    let ska_llm_lib_path = Path::new(&ska_llm_lib_path_str);
    let contents = fs::read_to_string(
        global_state.env_config.ska_data_dir.clone()
            + "/thapo_ska_py/ska_llm/scripts/download_llms.py",
    )?;

    println!("{}", contents);
    tracing::trace!("action_download.do_download end");
    Ok(())
}
