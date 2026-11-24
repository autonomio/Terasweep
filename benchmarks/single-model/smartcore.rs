use std::error::Error;
use std::time::Instant;

use serde_json::json;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{
    RidgeRegression, RidgeRegressionParameters, RidgeRegressionSolverName,
};
struct Data {
    rows: usize,
    cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn read_matrix(bytes: &[u8]) -> Result<Data, Box<dyn Error>> {
    let header = bytes.get(..24).ok_or("Truncated matrix header")?;
    if &header[..8] != b"RIDGE001" {
        return Err("Invalid matrix magic".into());
    }
    let rows = usize::try_from(u64::from_le_bytes(header[8..16].try_into()?))?;
    let cols = usize::try_from(u64::from_le_bytes(header[16..24].try_into()?))?;
    let values = rows.checked_mul(cols).ok_or("Matrix size overflow")?;
    let expected = values
        .checked_add(rows)
        .and_then(|n| n.checked_mul(8))
        .ok_or("Matrix size overflow")?;
    if bytes.len() - 24 != expected {
        return Err("Invalid matrix length".into());
    }
    let values_iter = bytes[24..]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()));
    let mut x: Vec<_> = values_iter.collect();
    let y = x.split_off(values);
    Ok(Data { rows, cols, x, y })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let train = read_matrix(&std::fs::read(args.next().ok_or("Missing train file")?)?)?;
    let test = read_matrix(&std::fs::read(args.next().ok_or("Missing test file")?)?)?;
    let alpha: f64 = args
        .next()
        .and_then(|x| x.into_string().ok())
        .ok_or("Missing alpha")?
        .parse()?;
    let start = Instant::now();
    let x_train = DenseMatrix::new(train.rows, train.cols, train.x, false)?;
    let x_test = DenseMatrix::new(test.rows, test.cols, test.x, false)?;
    let model = RidgeRegression::fit(
        &x_train,
        &train.y,
        RidgeRegressionParameters::default()
            .with_alpha(alpha)
            .with_normalize(false)
            .with_solver(RidgeRegressionSolverName::Cholesky),
    )?;
    let predictions = model.predict(&x_test)?;
    let fit_predict_seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"predictions": predictions, "fit_predict_seconds": fit_predict_seconds})
    );
    Ok(())
}
