use std::env;
use std::process::Command;
use std::{fs, path::Path};

use pyo3::types::PyList;

use crate::domain::nn_model::{service_nn_model, NnModelData};
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;
use crate::prelude::*;

pub fn do_download_data(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_download start");
    let download_dir = my_paths::get_models_download_data_dir(&global_state.env_config);
    let download_dir = download_dir.as_path();
    if download_dir.exists() {
        fs::remove_dir_all(download_dir)?;
    }
    fs::create_dir_all(download_dir)?;

    env::set_current_dir(download_dir)?;
    wget_download_website("https://www.w3schools.com/")?;
    wget_download_website("https://en.wikipedia.org/")?;
    tracing::trace!("do_download end");
    Ok(())
}

// fn py_download_all_data(contents: String, download_dir: &Path) -> anyhow::Result<()> {
//     pyo3::prepare_freethreaded_python();
//     let arg1 = download_dir.to_str().ok_or(anyhow::anyhow!(
//         "download_dir could not converted to string"
//     ))?;

//     Python::with_gil(|py| {
//         let fun: Py<PyAny> = PyModule::from_code_bound(py, contents.as_str(), "", "")?
//             .getattr("download_all_data")?
//             .into();

//         // pass arguments as rust tuple
//         // let nn_modesl_list = PyList::new(py, nn_models.to_vec());
//         // let arg2: [NnModelData; 4] = nn_models.try_into().unwrap();
//         // let arg2: Vec<(String, String, String, String)> = nn_models
//         //     .iter()
//         //     .map(|nn_model| nn_model.into_tuple())
//         //     .collect();
//         let args = (arg1,);
//         fun.call1(py, args)?;
//         Ok(())
//     })
// }

fn wget_download_website(website: &str) -> anyhow::Result<()> {
    Command::new("wget")
        .args(["-m", "-p", "-E", "-k", website])
        .spawn()?;
    Ok(())
}
