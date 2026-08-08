//! Token-embedding lookup for flat row-major FP32 tables.

use crate::error::{EngineError, Result};
use crate::tensor::{checked_matrix_len, matrix_row, validate_matrix};

/// Gather token embeddings from a table shaped `[vocab_size, hidden_size]`.
///
/// The returned vector is row-major with shape
/// `[token_ids.len(), hidden_size]`.
///
/// # Errors
///
/// Returns [`EngineError::InvalidInput`] for invalid dimensions, an incorrectly
/// sized table, an empty token sequence, or an out-of-vocabulary token.
pub fn embedding_lookup(
    token_ids: &[u32],
    table: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        return Err(EngineError::invalid_input(
            "embedding lookup",
            "at least one token id is required",
        ));
    }

    validate_matrix("embedding table", table, vocab_size, hidden_size)?;
    let output_len = checked_matrix_len("embedding output", token_ids.len(), hidden_size)?;
    let mut output = Vec::with_capacity(output_len);

    for &token_id in token_ids {
        let row = usize::try_from(token_id).map_err(|_| {
            EngineError::invalid_input(
                "embedding lookup",
                format!("token id {token_id} cannot be represented as usize"),
            )
        })?;
        if row >= vocab_size {
            return Err(EngineError::invalid_input(
                "embedding lookup",
                format!("token id {token_id} is outside vocabulary size {vocab_size}"),
            ));
        }

        output.extend_from_slice(matrix_row(
            "embedding table",
            table,
            vocab_size,
            hidden_size,
            row,
        )?);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::embedding_lookup;

    #[test]
    fn gathers_rows_in_token_order() {
        let table = [
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ];
        let output = embedding_lookup(&[2, 0], &table, 3, 2).expect("valid lookup");
        assert_eq!(output, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_out_of_vocabulary_tokens() {
        let error =
            embedding_lookup(&[3], &[0.0; 6], 3, 2).expect_err("token id 3 must be rejected");
        assert!(error.to_string().contains("outside vocabulary"));
    }
}
