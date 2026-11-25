use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::{FitPrediction, Matrix};

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub lookback: usize,
    pub hidden_width: usize,
    pub learning_rate: f64,
    pub epochs: usize,
}

pub(crate) fn fit_batch(
    train: &Matrix,
    test: &Matrix,
    configs: &[TrainConfig],
    seed: u32,
) -> Result<Vec<FitPrediction>, Box<dyn Error>> {
    let python = std::env::var_os("TERASWEEP_TIDE_PYTHON")
        .ok_or("Set TERASWEEP_TIDE_PYTHON to a Python interpreter with TensorFlow 2.16.2")?;
    if std::env::var_os("TERASWEEP_TIDE_SOURCE").is_none() {
        return Err("Set TERASWEEP_TIDE_SOURCE to the pinned Google Research models.py".into());
    }
    let mut child = Command::new(python)
        .args(["-B", "-c", include_str!("../scripts/tide_backend.py")])
        .env("TF_USE_LEGACY_KERAS", "0")
        .env("OMP_NUM_THREADS", "1")
        .env("TF_NUM_INTEROP_THREADS", "1")
        .env("TF_NUM_INTRAOP_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let sent = (|| -> std::io::Result<()> {
        let mut input = std::io::BufWriter::new(child.stdin.take().unwrap());
        write!(input, "[{},{},{},[", seed, train.rows, test.rows)?;
        for (index, c) in configs.iter().enumerate() {
            if index > 0 {
                write!(input, ",")?;
            }
            write!(
                input,
                "[{},{},{},{}]",
                c.lookback, c.hidden_width, c.learning_rate, c.epochs
            )?;
        }
        writeln!(input, "]]")?;
        for value in train.x.iter().chain(&train.y).chain(&test.x) {
            input.write_all(&value.to_le_bytes())?;
        }
        input.flush()
    })();
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("TiDE reference backend failed: {}", output.status).into());
    }
    sent?;
    let row_bytes = test
        .rows
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or("TiDE prediction size overflow")?;
    if row_bytes.checked_mul(configs.len()) != Some(output.stdout.len()) {
        return Err("TiDE reference backend returned an unexpected prediction count".into());
    }
    output
        .stdout
        .chunks_exact(row_bytes)
        .map(|data| {
            let fit_predict_seconds = f64::from_le_bytes(data[..8].try_into()?);
            let predictions = data[8..]
                .chunks_exact(8)
                .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            Ok(FitPrediction {
                predictions,
                fit_predict_seconds,
            })
        })
        .collect()
}
