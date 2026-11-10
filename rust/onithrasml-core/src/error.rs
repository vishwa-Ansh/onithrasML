
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OnithrasError {
    #[error("Input dimensions do not match")]
    DimensionMismatch,

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model has not been fitted")]
    ModelNotFitted,
}

pub type Result<T> = std::result::Result<T, OnithrasError>;
