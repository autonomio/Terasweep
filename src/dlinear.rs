use crate::Matrix;

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub alpha: f64,
    pub solver_eps: f64,
    pub lookback: usize,
    pub kernel: usize,
}

struct DesignStats {
    dims: usize,
    x_mean: Vec<f64>,
    y_mean: f64,
    centered_gram: Vec<f64>,
    centered_rhs: Vec<f64>,
}

struct FeatureMatrix {
    rows: usize,
    dims: usize,
    x: Vec<f64>,
}

pub(crate) struct DesignCache {
    train: DesignStats,
    test: FeatureMatrix,
}

impl DesignCache {
    pub(crate) fn new(
        train: &Matrix,
        test: &Matrix,
        lookback: usize,
        kernel: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let valid_size = |elements: Option<usize>| {
            elements.map_or(false, |n| {
                n <= isize::MAX as usize / std::mem::size_of::<f64>()
            })
        };
        let dims = lookback
            .checked_mul(2)
            .ok_or("DLinear lookback is too large")?;
        if !valid_size(dims.checked_mul(dims))
            || !valid_size(dims.checked_mul(test.rows))
            || !lookback
                .checked_add(kernel)
                .map_or(false, |n| n <= isize::MAX as usize)
        {
            return Err("DLinear window dimensions are too large".into());
        }
        Ok(Self {
            train: build_design_stats(train, lookback, kernel),
            test: build_feature_matrix(test, lookback, kernel),
        })
    }

    pub(crate) fn fit_predict(
        &self,
        config: TrainConfig,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let beta = fit_from_stats(&self.train, config)?;
        Ok(predict(&self.test, &beta))
    }
}

fn build_design_stats(data: &Matrix, lookback: usize, kernel: usize) -> DesignStats {
    let dims = 2 * lookback * data.cols;
    let mut x_mean = vec![0.0; dims];
    let mut y_mean = 0.0;
    let mut row_features = vec![0.0; dims];
    let inv_rows = 1.0 / data.rows as f64;

    for row in 0..data.rows {
        fill_features(data, row, lookback, kernel, &mut row_features);
        y_mean += data.y[row];
        for col in 0..dims {
            x_mean[col] += row_features[col];
        }
    }
    y_mean *= inv_rows;
    for value in &mut x_mean {
        *value *= inv_rows;
    }

    let mut centered_gram = vec![0.0; dims * dims];
    let mut centered_rhs = vec![0.0; dims];

    for row in 0..data.rows {
        fill_features(data, row, lookback, kernel, &mut row_features);
        let yc = data.y[row] - y_mean;
        for i in 0..dims {
            let xi_raw = row_features[i];
            let xi = xi_raw - x_mean[i];
            centered_rhs[i] += xi * yc;
            for j in 0..=i {
                centered_gram[i * dims + j] += xi * (row_features[j] - x_mean[j]);
            }
        }
    }

    DesignStats {
        dims,
        x_mean,
        y_mean,
        centered_gram,
        centered_rhs,
    }
}

fn build_feature_matrix(data: &Matrix, lookback: usize, kernel: usize) -> FeatureMatrix {
    let dims = 2 * lookback * data.cols;
    let mut x = vec![0.0; data.rows * dims];
    for row in 0..data.rows {
        fill_features(
            data,
            row,
            lookback,
            kernel,
            &mut x[row * dims..(row + 1) * dims],
        );
    }
    FeatureMatrix {
        rows: data.rows,
        dims,
        x,
    }
}

fn fill_features(data: &Matrix, row: usize, lookback: usize, kernel: usize, out: &mut [f64]) {
    let cols = data.cols;
    let trend_offset = lookback * cols;
    for pos in 0..lookback {
        let row_idx = window_row(row, pos, lookback);
        let source = row_idx * cols;
        for col in 0..cols {
            let raw = data.x[source + col];
            let trend = moving_average(data, row, pos, col, lookback, kernel);
            let idx = pos * cols + col;
            out[idx] = raw - trend;
            out[trend_offset + idx] = trend;
        }
    }
}

fn moving_average(
    data: &Matrix,
    row: usize,
    pos: usize,
    col: usize,
    lookback: usize,
    kernel: usize,
) -> f64 {
    let half = kernel / 2;
    let mut sum = 0.0;
    for k in 0..kernel {
        let shifted = pos as isize + k as isize - half as isize;
        let clamped = shifted.clamp(0, lookback as isize - 1) as usize;
        let row_idx = window_row(row, clamped, lookback);
        sum += data.x[row_idx * data.cols + col];
    }
    sum / kernel as f64
}

fn window_row(row: usize, pos: usize, lookback: usize) -> usize {
    let available = (row + 1).min(lookback);
    let pad = lookback - available;
    if pos < pad {
        0
    } else {
        row + 1 - available + pos - pad
    }
}

fn fit_from_stats(
    stats: &DesignStats,
    config: TrainConfig,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let dims = stats.dims;
    let mut gram = stats.centered_gram.clone();
    let mut rhs = stats.centered_rhs.clone();

    for i in 0..dims {
        gram[i * dims + i] += config.alpha;
        for j in 0..i {
            gram[j * dims + i] = gram[i * dims + j];
        }
    }

    solve_spd_in_place(&mut gram, &mut rhs, dims, config.solver_eps)?;
    let intercept = stats.y_mean
        - stats
            .x_mean
            .iter()
            .zip(rhs.iter())
            .map(|(x, b)| x * b)
            .sum::<f64>();
    let mut beta = Vec::with_capacity(dims + 1);
    beta.push(intercept);
    beta.extend_from_slice(&rhs);
    Ok(beta)
}

fn predict(features: &FeatureMatrix, beta: &[f64]) -> Vec<f64> {
    let mut predictions = Vec::with_capacity(features.rows);
    for row in features.x.chunks_exact(features.dims) {
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
                    return Err(format!("singular dlinear matrix at pivot {row}: {sum}").into());
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
