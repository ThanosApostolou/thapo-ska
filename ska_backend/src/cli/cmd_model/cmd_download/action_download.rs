use std::{fs, path::Path};

use pyo3::types::PyList;

use crate::domain::nn_model::{service_nn_model, NnModelData};
use crate::modules::global_state::GlobalState;
use crate::modules::myfs::my_paths;
use crate::prelude::*;

pub fn do_download(global_state: &GlobalState) -> anyhow::Result<()> {
    tracing::trace!("do_download start");
    let contents = fs::read_to_string(global_state.env_config.ska_llm_dir.clone() + "/lib.py")
        .expect("Should have been able to read the file");

    let download_dir = my_paths::get_models_download_dir(&global_state.env_config);
    let download_dir = download_dir.as_path();
    if download_dir.exists() {
        fs::remove_dir_all(download_dir)?;
    }
    fs::create_dir_all(download_dir)?;

    println!("{}", contents);

    let nn_models = service_nn_model::get_nn_models_list();
    python_test(contents, download_dir, nn_models).unwrap();
    tracing::trace!("do_download end");
    Ok(())
}

fn python_test(contents: String, download_dir: &Path, nn_models: Vec<NnModelData>) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    let arg1 = download_dir.to_str().unwrap();

    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code_bound(py, contents.as_str(), "", "")?
            .getattr("download_llm_huggingface")?
            .into();

        // pass arguments as rust tuple
        // let nn_modesl_list = PyList::new(py, nn_models.to_vec());
        // let arg2: [NnModelData; 4] = nn_models.try_into().unwrap();
        let arg2: Vec<(String, String, String, String)> = nn_models
            .iter()
            .map(|nn_model| nn_model.into_tuple())
            .collect();
        let args = (arg1, arg2);
        fun.call1(py, args)?;
        Ok(())
    })
}
