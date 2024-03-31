use pyo3::prelude::*;
use pyo3::types::PyTuple;
use ska_backend::modules::global_state::GlobalState;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let secret_file =
        std::env::var("THAPO_SKA_SECRET_FILE").unwrap_or_else(|_| ".secret".to_string());
    dotenv::from_filename(&secret_file)
        .unwrap_or_else(|_| panic!("could not load file {}", secret_file.clone()));

    let global_state = GlobalState::initialize_cli().await.unwrap();
    let global_state = Arc::new(global_state);

    println!("hello from cli");
    let ska_llm_lib_path_str =
        global_state.env_config.ska_data_dir.clone() + "/thapo_ska_py/ska_llm/scripts/download_llms.py";
    tracing::info!("path_str: {}", ska_llm_lib_path_str);
    let ska_llm_lib_path = Path::new(&ska_llm_lib_path_str);
    let contents = fs::read_to_string(
        global_state.env_config.ska_data_dir.clone() + "/thapo_ska_py/ska_llm/scripts/download_llms.py",
    )
    .expect("Should have been able to read the file");
    println!("{}", contents);
    python_test(contents).unwrap();
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
