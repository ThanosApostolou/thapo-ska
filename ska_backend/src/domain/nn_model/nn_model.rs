#[derive(Debug, Clone)]
pub struct NnModelData {
    pub repo_id: String,
    pub rel_path: String,
    pub revision: String,
    pub allow_patterns: String,
}

impl NnModelData {
    pub fn into_tuple(&self) -> (String, String, String, String) {
        (
            self.repo_id.clone(),
            self.rel_path.clone(),
            self.revision.clone(),
            self.allow_patterns.clone(),
        )
    }
}
