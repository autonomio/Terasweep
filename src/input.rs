use crate::results::Prices;

const MATRIX_MAGIC: &[u8; 8] = b"RIDGE001";
const PRICE_MAGIC: &[u8; 8] = b"PRICE001";
const F64_BYTES: usize = 8;

/// A prepared feature matrix with row-major `x` and one target per row.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Matrix {
    pub(crate) fn has_shape(&self) -> bool {
        self.rows > 0
            && self.cols > 0
            && self.rows.checked_mul(self.cols) == Some(self.x.len())
            && self.y.len() == self.rows
    }
}

/// Decode a Terasweep `RIDGE001` matrix.
pub fn read_matrix(bytes: &[u8]) -> Result<Matrix, Box<dyn std::error::Error>> {
    if bytes.len() < 24 {
        return Err("matrix file is too short".into());
    }
    if &bytes[0..8] != MATRIX_MAGIC {
        return Err("invalid ridge data magic".into());
    }

    let rows = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into()?))?;
    let cols = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into()?))?;
    let x_count = rows.checked_mul(cols).ok_or("rows*cols overflow")?;
    let expected_values = x_count.checked_add(rows).ok_or("value count overflow")?;
    let expected = expected_values
        .checked_mul(F64_BYTES)
        .and_then(|n| n.checked_add(24))
        .ok_or("matrix byte count overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "matrix size {} does not match expected {}",
            bytes.len(),
            expected
        )
        .into());
    }

    let mut offset = 24;
    let mut x = Vec::with_capacity(x_count);
    for _ in 0..x_count {
        x.push(read_f64(bytes, &mut offset));
    }
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        y.push(read_f64(bytes, &mut offset));
    }

    Ok(Matrix { rows, cols, x, y })
}

/// Decode the opening and closing price arrays in a `PRICE001` file.
pub fn read_prices(bytes: &[u8]) -> Result<Prices, Box<dyn std::error::Error>> {
    if bytes.len() < 16 {
        return Err("price file is too short".into());
    }
    if &bytes[0..8] != PRICE_MAGIC {
        return Err("invalid price data magic".into());
    }

    let rows = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into()?))?;
    let expected = rows
        .checked_mul(2 * F64_BYTES)
        .and_then(|n| n.checked_add(16))
        .ok_or("price byte count overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "price size {} does not match expected {}",
            bytes.len(),
            expected
        )
        .into());
    }

    let mut offset = 16;
    let mut open = Vec::with_capacity(rows);
    for _ in 0..rows {
        open.push(read_f64(bytes, &mut offset));
    }
    let mut close = Vec::with_capacity(rows);
    for _ in 0..rows {
        close.push(read_f64(bytes, &mut offset));
    }

    Ok(Prices { open, close })
}

fn read_f64(bytes: &[u8], offset: &mut usize) -> f64 {
    let value = f64::from_le_bytes(bytes[*offset..*offset + F64_BYTES].try_into().unwrap());
    *offset += F64_BYTES;
    value
}
