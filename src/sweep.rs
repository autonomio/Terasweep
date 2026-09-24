mod dlinear;
mod lightgbm;
mod results;
mod ridge;
mod tide;

use results::Prices;
use std::env;
use std::fs;
use std::time::Instant;

const MATRIX_MAGIC: &[u8; 8] = b"RIDGE001";
const PRICE_MAGIC: &[u8; 8] = b"PRICE001";
const F64_BYTES: usize = 8;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: terasweep <train.bin> <test.bin> <price.bin> <config.json> <runs> <out.json>"
                .into(),
        );
    }

    let train = read_matrix(&fs::read(&args[1])?)?;
    let test = read_matrix(&fs::read(&args[2])?)?;
    let prices = read_prices(&fs::read(&args[3])?)?;
    let config_json = fs::read_to_string(&args[4])?;
    let runs: usize = args[5].parse()?;

    if test.cols != train.cols {
        return Err(format!("test cols {} != train cols {}", test.cols, train.cols).into());
    }
    if prices.open.len() != test.rows {
        return Err(format!(
            "price rows {} != test rows {}",
            prices.open.len(),
            test.rows
        )
        .into());
    }

    let price_stats = results::precompute_price_stats(&prices);
    let hot_start = Instant::now();
    let run = match results::model_name(&config_json)?
        .to_ascii_lowercase()
        .as_str()
    {
        "ridge" => ridge::run(&config_json, &train, &test, &price_stats, runs)?,
        "dlinear" => dlinear::run(&config_json, &train, &test, &price_stats, runs)?,
        "lightgbm" => lightgbm::run(&config_json, &train, &test, &price_stats, runs)?,
        "tide" => tide::run(&config_json, &train, &test, &price_stats, runs)?,
        other => return Err(format!("bad model value: {other}").into()),
    };
    let hot_seconds = hot_start.elapsed().as_secs_f64();

    results::write_summary_json(&args[6], &run, hot_seconds)?;

    Ok(())
}

fn read_matrix(bytes: &[u8]) -> Result<Matrix, Box<dyn std::error::Error>> {
    if bytes.len() < 24 {
        return Err("matrix file is too short".into());
    }
    if &bytes[0..8] != MATRIX_MAGIC {
        return Err("invalid ridge data magic".into());
    }

    let rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let cols = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let x_count = rows.checked_mul(cols).ok_or("rows*cols overflow")?;
    let expected_values = x_count.checked_add(rows).ok_or("value count overflow")?;
    let expected = 24 + expected_values * F64_BYTES;
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

fn read_prices(bytes: &[u8]) -> Result<Prices, Box<dyn std::error::Error>> {
    if bytes.len() < 16 {
        return Err("price file is too short".into());
    }
    if &bytes[0..8] != PRICE_MAGIC {
        return Err("invalid price data magic".into());
    }

    let rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let expected = 16 + rows * F64_BYTES * 2;
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
