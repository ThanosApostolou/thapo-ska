use std::{
    env,
    fs::{self, DirEntry},
    io,
    path::{Path, PathBuf},
};

pub fn get_up_files(migration_dir: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    let data_dir = env::var("THAPO_SKA_DATA_DIR").unwrap();
    let up_dir = PathBuf::from(&data_dir)
        .join("migration")
        .join(migration_dir)
        .join("up");

    print!("up_dir={}", up_dir.to_str().unwrap());

    let mut paths: Vec<_> = fs::read_dir(up_dir).unwrap().map(|r| r.unwrap()).collect();
    paths.sort_by_key(|dir| dir.file_name());

    Ok(paths)
}

pub fn get_down_files(migration_dir: impl AsRef<Path>) -> io::Result<Vec<DirEntry>> {
    let data_dir = env::var("THAPO_SKA_DATA_DIR").unwrap();
    let down_dir = PathBuf::from(&data_dir)
        .join("migration")
        .join(migration_dir)
        .join("down");

    let mut paths: Vec<_> = fs::read_dir(down_dir)
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    paths.sort_by_key(|dir| dir.file_name());

    Ok(paths)
}
