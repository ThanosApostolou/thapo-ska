use std::{fs, path::Path};

use crate::modules::global_state::GlobalState;
use crate::prelude::*;

pub fn do_download(global_state: &GlobalState) {
    tracing::trace!("action_download.do_download start");
    let ska_llm_lib_path_str = global_state.env_config.ska_data_dir.clone()
        + "/thapo_ska_py/ska_llm/scripts/download_llms.py";
    tracing::info!("path_str: {}", ska_llm_lib_path_str);
    let ska_llm_lib_path = Path::new(&ska_llm_lib_path_str);
    let contents = fs::read_to_string(
        global_state.env_config.ska_data_dir.clone()
            + "/thapo_ska_py/ska_llm/scripts/download_llms.py",
    )
    .expect("Should have been able to read the file");

    println!("{}", contents);
    python_test(contents).unwrap();
    tracing::trace!("action_download.do_download end");
}

fn python_test(contents: String) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    let arg1 = "arg1";
    let arg2 = "arg2";
    let arg3 = "arg3";

    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code_bound(py, contents.as_str(), "", "")?
            .getattr("main")?
            .into();

        // call object without any arguments
        fun.call0(py)?;

        // call object with PyTuple
        // let args = PyTuple::new(py, &[arg1, arg2, arg3]);
        // fun.call1(py, args)?;

        // pass arguments as rust tuple
        // let args = (arg1, arg2, arg3);
        // fun.call1(py, args)?;
        Ok(())
    })
}
