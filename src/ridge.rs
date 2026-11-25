use crate::Matrix;

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub alpha: f64,
    pub fit_intercept: bool,
    pub solver_eps: f64,
}

pub struct RidgeStats {
    cols: usize,
    x_mean: Vec<f64>,
    y_mean: f64,
    centered_gram: Vec<f64>,
    centered_rhs: Vec<f64>,
    raw_gram: Vec<f64>,
    raw_rhs: Vec<f64>,
}

pub fn precompute_stats(data: &Matrix) -> RidgeStats {
    let rows = data.rows;
    let cols = data.cols;
    let inv_rows = 1.0 / rows as f64;
    let mut x_mean = vec![0.0; cols];
    let mut y_mean = 0.0;

    for row in 0..rows {
        let offset = row * cols;
        y_mean += data.y[row];
        for col in 0..cols {
            x_mean[col] += data.x[offset + col];
        }
    }
    y_mean *= inv_rows;
    for value in &mut x_mean {
        *value *= inv_rows;
    }

    let mut centered_gram = vec![0.0; cols * cols];
    let mut centered_rhs = vec![0.0; cols];
    let mut raw_gram = vec![0.0; cols * cols];
    let mut raw_rhs = vec![0.0; cols];
    for row in 0..rows {
        let offset = row * cols;
        let yc = data.y[row] - y_mean;
        for i in 0..cols {
            let xi_raw = data.x[offset + i];
            let xi = data.x[offset + i] - x_mean[i];
            centered_rhs[i] += xi * yc;
            raw_rhs[i] += xi_raw * data.y[row];
            for j in 0..=i {
                centered_gram[i * cols + j] += xi * (data.x[offset + j] - x_mean[j]);
                raw_gram[i * cols + j] += xi_raw * data.x[offset + j];
            }
        }
    }

    RidgeStats {
        cols,
        x_mean,
        y_mean,
        centered_gram,
        centered_rhs,
        raw_gram,
        raw_rhs,
    }
}

pub(crate) fn fit_predict(
    stats: &RidgeStats,
    test: &Matrix,
    config: TrainConfig,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let beta = fit_from_stats(stats, config)?;
    Ok(predict(test, &beta))
}

fn fit_from_stats(
    stats: &RidgeStats,
    config: TrainConfig,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let cols = stats.cols;
    let mut gram = if config.fit_intercept {
        stats.centered_gram.clone()
    } else {
        stats.raw_gram.clone()
    };
    let mut rhs = if config.fit_intercept {
        stats.centered_rhs.clone()
    } else {
        stats.raw_rhs.clone()
    };

    for i in 0..cols {
        gram[i * cols + i] += config.alpha;
        for j in 0..i {
            gram[j * cols + i] = gram[i * cols + j];
        }
    }

    solve_spd_in_place(&mut gram, &mut rhs, cols, config.solver_eps)?;
    let intercept = if config.fit_intercept {
        stats.y_mean
            - stats
                .x_mean
                .iter()
                .zip(rhs.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
    } else {
        0.0
    };
    let mut beta = Vec::with_capacity(cols + 1);
    beta.push(intercept);
    beta.extend_from_slice(&rhs);
    Ok(beta)
}

fn predict(data: &Matrix, beta: &[f64]) -> Vec<f64> {
    let mut predictions = Vec::with_capacity(data.rows);
    for row in data.x.chunks_exact(data.cols) {
        let mut prediction = beta[0];
        for (value, coefficient) in row.iter().zip(&beta[1..]) {
            prediction += value * coefficient;
        }
        predictions.push(prediction);
    }
    predictions
}

fn solve_spd_in_place(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    eps: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    for row in 0..n {
        let row_offset = row * n;
        for col in 0..=row {
            let col_offset = col * n;
            let mut sum = a[row_offset + col];
            for k in 0..col {
                sum -= a[row_offset + k] * a[col_offset + k];
            }

            if row == col {
                if sum <= eps {
                    return Err(format!("singular ridge matrix at pivot {row}: {sum}").into());
                }
                a[row_offset + col] = sum.sqrt();
            } else {
                a[row_offset + col] = sum / a[col_offset + col];
            }
        }
    }

    for row in 0..n {
        let row_offset = row * n;
        let mut value = b[row];
        for col in 0..row {
            value -= a[row_offset + col] * b[col];
        }
        b[row] = value / a[row_offset + row];
    }

    for row in (0..n).rev() {
        let row_offset = row * n;
        let mut value = b[row];
        for col in (row + 1)..n {
            value -= a[col * n + row] * b[col];
        }
        b[row] = value / a[row_offset + row];
    }

    Ok(())
}
