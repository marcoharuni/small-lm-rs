//! Checked helpers for flat row-major CPU tensors.

use crate::error::{EngineError, Result};

/// Compute the number of values in a matrix after validating its dimensions.
pub(crate) fn checked_matrix_len(
    component: &'static str,
    rows: usize,
    columns: usize,
) -> Result<usize> {
    if rows == 0 || columns == 0 {
        return Err(EngineError::invalid_input(
            component,
            "matrix dimensions must be greater than zero",
        ));
    }

    rows.checked_mul(columns)
        .ok_or_else(|| EngineError::invalid_input(component, "matrix shape overflows usize"))
}

/// Validate a flat slice against a row-major matrix shape.
pub(crate) fn validate_matrix(
    component: &'static str,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Result<()> {
    let expected = checked_matrix_len(component, rows, columns)?;
    if values.len() != expected {
        return Err(EngineError::invalid_input(
            component,
            format!(
                "received {} values, expected {expected} for shape [{rows}, {columns}]",
                values.len()
            ),
        ));
    }
    Ok(())
}

/// Return one row from a validated flat row-major matrix.
pub(crate) fn matrix_row<'a>(
    component: &'static str,
    values: &'a [f32],
    rows: usize,
    columns: usize,
    row: usize,
) -> Result<&'a [f32]> {
    validate_matrix(component, values, rows, columns)?;
    if row >= rows {
        return Err(EngineError::invalid_input(
            component,
            format!("row index {row} is outside 0..{rows}"),
        ));
    }

    let start = row
        .checked_mul(columns)
        .ok_or_else(|| EngineError::invalid_input(component, "row offset overflows usize"))?;
    Ok(&values[start..start + columns])
}

#[cfg(test)]
mod tests {
    use super::{checked_matrix_len, matrix_row, validate_matrix};

    #[test]
    fn checked_matrix_length_rejects_zero_and_overflow() {
        assert!(checked_matrix_len("test", 0, 4).is_err());
        assert!(checked_matrix_len("test", usize::MAX, 2).is_err());
    }

    #[test]
    fn row_major_validation_and_indexing_work() {
        let values = [1.0, 2.0, 3.0, 4.0];
        validate_matrix("test", &values, 2, 2).expect("valid matrix");
        assert_eq!(
            matrix_row("test", &values, 2, 2, 1).expect("second row"),
            &[3.0, 4.0]
        );
        assert!(matrix_row("test", &values, 2, 2, 2).is_err());
    }
}
