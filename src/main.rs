use std::{env, fs};
use terasweep::{read_matrix, read_prices, sweep};

mod single_fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map_or(false, |arg| arg == "_fit") {
        return single_fit::command(&args[2..]);
    }
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

    let result = sweep(&config_json, &train, &test, &prices, runs)?;
    result.write_json(fs::File::create(&args[6])?)?;
    Ok(())
}
