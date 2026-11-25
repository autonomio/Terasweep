use std::error::Error;
use std::io::{self, Write};
use std::{env, fs};

use terasweep::{read_matrix, sweep};

mod single_fit;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [command, tail @ ..] if command == "_fit" => single_fit::command(tail),
        [command, train, validation, plan] if command == "sweep" => {
            sweep_command(train, validation, plan)
        }
        _ => Err("usage: terasweep sweep <train.bin> <validation.bin> <models.txt>".into()),
    }
}

fn sweep_command(train: &str, validation: &str, plan: &str) -> Result<(), Box<dyn Error>> {
    let train = read_matrix(&fs::read(train)?)?;
    let validation = read_matrix(&fs::read(validation)?)?;
    let mut models = Vec::new();
    for (line, text) in fs::read_to_string(plan)?.lines().enumerate() {
        let text = text.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let settings: Vec<_> = text.split_whitespace().collect();
        let model = single_fit::parse_model(&settings)
            .map_err(|error| format!("{plan}:{}: {error}", line + 1))?;
        models.push(model);
    }
    let result = sweep(&models, &train, &validation)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(
        out,
        "{{\"trials\":{},\"unique_fits\":{},\"elapsed_seconds\":{},\"best\":{{\"trial\":{},\"mae\":{},\"predictions\":{:?}}}}}",
        result.trials,
        result.unique_fits,
        result.elapsed_seconds,
        result.best.trial,
        result.best.mae,
        result.best.predictions,
    )?;
    out.flush()?;
    Ok(())
}
