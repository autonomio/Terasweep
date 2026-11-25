use std::error::Error;
use std::io::{self, Read, Write};

use terasweep::{fit_predict, read_matrix, Model};

pub(crate) fn command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let model = parse_model(&args)?;
    let mut bytes = Vec::new();
    io::stdin().lock().read_to_end(&mut bytes)?;
    let header = bytes.get(..8).ok_or("Missing fit payload header")?;
    let train_bytes = usize::try_from(u64::from_le_bytes(header.try_into()?))?;
    let payload = &bytes[8..];
    let train = read_matrix(payload.get(..train_bytes).ok_or("Truncated train matrix")?)?;
    let test = read_matrix(&payload[train_bytes..])?;
    let fit = fit_predict(&model, &train, &test)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    out.write_all(&fit.fit_predict_seconds.to_le_bytes())?;
    for prediction in fit.predictions {
        out.write_all(&prediction.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

pub(crate) fn parse_model(args: &[&str]) -> Result<Model, Box<dyn Error>> {
    Ok(match args {
        ["ridge", alpha, intercept, eps] => Model::Ridge {
            alpha: alpha.parse()?,
            fit_intercept: intercept.parse()?,
            solver_eps: eps.parse()?,
        },
        ["dlinear", alpha, lookback, kernel, eps] => Model::DLinear {
            alpha: alpha.parse()?,
            lookback: lookback.parse()?,
            kernel: kernel.parse()?,
            solver_eps: eps.parse()?,
        },
        ["lightgbm", rate, trees, leaves, bins, l2, min_leaf, seed] => Model::LightGbm {
            learning_rate: rate.parse()?,
            num_trees: trees.parse()?,
            num_leaves: leaves.parse()?,
            max_bins: bins.parse()?,
            lambda_l2: l2.parse()?,
            min_data_in_leaf: min_leaf.parse()?,
            seed: seed.parse()?,
        },
        ["tide", lookback, width, rate, epochs, seed] => Model::Tide {
            lookback: lookback.parse()?,
            hidden_width: width.parse()?,
            learning_rate: rate.parse()?,
            epochs: epochs.parse()?,
            seed: seed.parse()?,
        },
        _ => return Err("Invalid model settings".into()),
    })
}
