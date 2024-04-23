pub mod cli;
pub mod domain;
pub mod modules;
pub mod server;

pub mod prelude {
    pub use sea_orm::prelude::*;
}
