use std::collections::{HashMap, HashSet};

use crate::results::{
    self, BaseScore, CommonSpace, PriceStats, RunOutput, ScoreConfig, ScoreKey, ScoredRow,
    SplitMix64, SweepSummary,
};
use crate::Matrix;
use crate::optimized::{self, Level};
use std::time::Instant;

const MAE_DECIMALS: i32 = 3;

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

pub struct FitOut {
    pub config: TrainConfig,
    pub preds: Vec<f64>,
    pub mae: f64,
    pub fit_failed: bool,
}

pub type TrainKey = (u64, bool, u64);

struct ModelSpace {
    alphas: Vec<f64>,
    fit_intercepts: Vec<bool>,
    solver_eps: Vec<f64>,
    plan: Option<Vec<Param>>,
}

#[derive(Clone, Copy)]
pub(crate) struct Param {
    pub(crate) id: usize,
    pub(crate) train: TrainConfig,
    pub(crate) score: ScoreConfig,
    pub(crate) train_ordinal: usize,
    pub(crate) threshold_ordinal: usize,
}

pub fn run(
    config_json: &str,
    train: &Matrix,
    test: &Matrix,
    price_stats: &PriceStats,
    runs: usize,
) -> Result<RunOutput, Box<dyn std::error::Error>> {
    let level=Level::from_env()?;
    let start=Instant::now();
    let common = CommonSpace::from_json(config_json)?;
    let model = ModelSpace::from_json(config_json)?;
    let train_configs = model.train_configs();
    if model.plan.as_ref().map_or(false, |p| p.len()!=runs) {return Err("plan row count does not match requested runs".into());}
    let uncached=matches!(level,Level::Naive|Level::Stats);
    let stats = if level==Level::Naive {None} else {Some(precompute_stats(train))};
    let stats_s=start.elapsed().as_secs_f64();
    let start=Instant::now();
    let mut train_cache = if uncached {
        let mut seen=HashSet::new();
        train_configs.iter().filter(|&&c|seen.insert(cache_key(c))).map(|&config|FitOut{config,preds:Vec::new(),mae:f64::NAN,fit_failed:false}).collect()
    } else {precompute_train_cache(stats.as_ref().unwrap(), test, &train_configs)?};
    let fit_s=start.elapsed().as_secs_f64();
    let mut visited=vec![!uncached;train_cache.len()];
    let train_index: HashMap<TrainKey, usize> = train_cache
        .iter()
        .enumerate()
        .map(|(idx, entry)| (cache_key(entry.config), idx))
        .collect();
    let ordinal_to_cache: Vec<usize>=train_configs.iter().map(|&c|train_index[&cache_key(c)]).collect();
    let mut score_cache: HashMap<ScoreKey, BaseScore> = HashMap::new();
    let mut summary = SweepSummary::new();

    let start=Instant::now();
    let mut bounds=optimized::OnlineBounds::new(train_cache.len());
    let score_cache_entries=if matches!(level,Level::Compact|Level::Intrinsics|Level::Assembly) {
        optimized::compact(level,runs,|id|model.sample(&common,id),&common,&ordinal_to_cache,&train_index,&train_cache,price_stats,&mut summary)?
    } else {
        for id in 0..runs {
            let param=model.sample(&common,id);
            if level==Level::Score {
                summary.record(evaluate(&param,&train_index,&train_cache,&mut score_cache,price_stats)?);
                continue;
            }
            let idx=train_index[&cache_key(param.train)];
            if uncached {
                let per_trial;
                let fit_stats=if level==Level::Naive {per_trial=precompute_stats(train);&per_trial} else {stats.as_ref().unwrap()};
                train_cache[idx]=precompute_train_cache(fit_stats,test,&[param.train])?.pop().unwrap();
                visited[idx]=true;
            }
            let entry=&train_cache[idx];
            let score=if entry.fit_failed {(f64::NAN,f64::NAN,f64::NAN)} else {
                let base=if level==Level::Bounds {
                    let key=ScoreKey{train_idx:idx,threshold_bits:param.score.threshold.to_bits(),rule_idx:param.score.signal_rule.index()};
                    *score_cache.entry(key).or_insert_with(||bounds.base(idx,&entry.preds,param.score,price_stats))
                } else {results::compute_base_score(&entry.preds,param.score.threshold,param.score.signal_rule,price_stats)};
                results::apply_costs(&base,entry.preds.len(),param.score)
            };
            summary.record(ScoredRow{id, signal_rule:param.score.signal_rule,fit_failed:entry.fit_failed,mae:results::round_to(entry.mae,3),signal_rate:score.0,net_return:score.1,sharpe:score.2});
        }
        score_cache.len()
    };
    eprintln!("PHASE level={level:?} stats={stats_s:.9} fit_predict={fit_s:.9} sweep={:.9}",start.elapsed().as_secs_f64());

    // Fill only unvisited configurations to preserve complete configured-pool
    // failure accounting. This does not replace any uncached trial fit.
    if visited.iter().any(|&x|!x) {
        let extra_stats;
        let fit_stats=match stats.as_ref() {Some(s)=>s,None=>{extra_stats=precompute_stats(train);&extra_stats}};
        for idx in 0..train_cache.len() {
            if !visited[idx] {train_cache[idx]=precompute_train_cache(fit_stats,test,&[train_cache[idx].config])?.pop().unwrap();}
        }
    }
    if let Some(path)=std::env::var_os("TERASWEEP_EXPORT_PREDICTIONS") {
        use std::io::Write;
        let mut out=std::io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(b"PRED0001")?;
        out.write_all(&(train_cache.len() as u64).to_le_bytes())?;
        for e in &train_cache {
            out.write_all(&e.config.alpha.to_le_bytes())?;
            out.write_all(&(e.config.fit_intercept as u64).to_le_bytes())?;
            out.write_all(&e.config.solver_eps.to_le_bytes())?;
            out.write_all(&(e.fit_failed as u64).to_le_bytes())?;
            out.write_all(&(e.preds.len() as u64).to_le_bytes())?;
            for value in &e.preds {out.write_all(&value.to_le_bytes())?;}
        }
        out.flush()?;
    }
    let best_return_json = row_json(
        summary.best_return(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    let best_sharpe_json = row_json(
        summary.best_sharpe(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    let best_mae_json = row_json(
        summary.best_mae(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    Ok(RunOutput {
        unique_alphas: model.alphas.len(),
        unique_train_configs: train_cache.len(),
        failed_train_configs: train_cache.iter().filter(|entry| entry.fit_failed).count(),
        score_cache_entries,
        best_return_json,
        best_sharpe_json,
        best_mae_json,
        summary,
    })
}

impl ModelSpace {
    fn from_json(src: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let alpha_section = results::json_object(src, "alphas")?;
        Ok(Self {
            alphas: results::logspace(
                results::json_f64(alpha_section, "log_min")?,
                results::json_f64(alpha_section, "log_max")?,
                results::json_usize(alpha_section, "count")?,
            ),
            fit_intercepts: results::json_bool_array(src, "fit_intercept")?,
            solver_eps: results::json_f64_array(src, "solver_eps")?,
            plan: read_plan()?,
        })
    }

    fn train_configs(&self) -> Vec<TrainConfig> {
        let mut configs = Vec::with_capacity(
            self.alphas.len() * self.fit_intercepts.len() * self.solver_eps.len(),
        );
        for &alpha in &self.alphas {
            for &fit_intercept in &self.fit_intercepts {
                for &solver_eps in &self.solver_eps {
                    configs.push(TrainConfig {
                        alpha,
                        fit_intercept,
                        solver_eps,
                    });
                }
            }
        }
        configs
    }

    fn sample(&self, common: &CommonSpace, id: usize) -> Param {
        if let Some(plan)=&self.plan {return plan[id];}
        let mut rng = SplitMix64::new(common.seed ^ results::mix_id(id as u64));
        let signal_rule = common.signal_rules[rng.index(common.signal_rules.len())];
        let thresholds = if signal_rule.is_abs() {
            &common.threshold_abs
        } else {
            &common.threshold_signed
        };
        let ai=rng.index(self.alphas.len());
        let ti=rng.index(thresholds.len());
        let fi=rng.index(self.fit_intercepts.len());
        let ei=rng.index(self.solver_eps.len());
        let alpha = self.alphas[ai];
        let threshold = thresholds[ti];
        let fit_intercept = self.fit_intercepts[fi];
        let solver_eps = self.solver_eps[ei];
        let fee_bps = common.fee_bps[rng.index(common.fee_bps.len())];
        let slippage_bps = common.slippage_bps[rng.index(common.slippage_bps.len())];
        Param {
            id,
            train_ordinal: (ai*self.fit_intercepts.len()+fi)*self.solver_eps.len()+ei,
            threshold_ordinal: ti,
            train: TrainConfig {
                alpha,
                fit_intercept,
                solver_eps,
            },
            score: ScoreConfig {
                threshold,
                fee_bps,
                slippage_bps,
                signal_rule,
            },
        }
    }
}

fn evaluate(
    param: &Param,
    train_index: &HashMap<TrainKey, usize>,
    train_cache: &[FitOut],
    score_cache: &mut HashMap<ScoreKey, BaseScore>,
    price_stats: &PriceStats,
) -> Result<ScoredRow, Box<dyn std::error::Error>> {
    let cache_idx = *train_index
        .get(&cache_key(param.train))
        .ok_or("missing cache entry")?;
    let entry = &train_cache[cache_idx];
    let score = if entry.fit_failed {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        results::score_cached(
            cache_idx,
            &entry.preds,
            param.score,
            score_cache,
            price_stats,
        )
    };
    Ok(ScoredRow {
        id: param.id,
        signal_rule: param.score.signal_rule,
        fit_failed: entry.fit_failed,
        mae: results::round_to(entry.mae, MAE_DECIMALS),
        signal_rate: score.0,
        net_return: score.1,
        sharpe: score.2,
    })
}

fn row_json(
    row: Option<ScoredRow>,
    model: &ModelSpace,
    common: &CommonSpace,
    train_index: &HashMap<TrainKey, usize>,
    train_cache: &[FitOut],
    price_stats: &PriceStats,
) -> String {
    match row {
        Some(row) => {
            let param = model.sample(common, row.id);
            let cache_idx = train_index[&cache_key(param.train)];
            let detail = results::compute_backtest_stats(
                &train_cache[cache_idx].preds,
                param.score,
                price_stats,
                common,
            );
            format!(
                "{{\"id\": {}, \"alpha\": {}, \"threshold\": {}, \"fit_intercept\": {}, \"solver_eps\": {}, \"fee_bps\": {}, \"slippage_bps\": {}, \"signal_rule\": \"{}\", \"mae\": {}, \"signal_rate_pct\": {}, \"backtest_total_return_net_pct\": {}, \"backtest_sharpe_per_bar\": {}, {}}}",
                row.id,
                results::json_float(param.train.alpha),
                results::json_float(param.score.threshold),
                param.train.fit_intercept,
                results::json_float(param.train.solver_eps),
                results::json_float(param.score.fee_bps),
                results::json_float(param.score.slippage_bps),
                param.score.signal_rule.as_str(),
                results::json_float(row.mae),
                results::json_float(row.signal_rate),
                results::json_float(row.net_return),
                results::json_float(row.sharpe),
                results::backtest_stats_json(detail)
            )
        }
        None => "null".to_string(),
    }
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

pub fn precompute_train_cache(
    stats: &RidgeStats,
    test: &Matrix,
    configs: &[TrainConfig],
) -> Result<Vec<FitOut>, Box<dyn std::error::Error>> {
    let mut cache: Vec<FitOut> = Vec::new();
    let mut seen: HashSet<TrainKey> = HashSet::new();
    for &config in configs {
        let key = cache_key(config);
        if !seen.insert(key) {
            continue;
        }
        match fit_from_stats(stats, config) {
            Ok(beta) => {
                let (preds, mae) = predict_and_mae(test, &beta);
                cache.push(FitOut {
                    config,
                    preds,
                    mae,
                    fit_failed: false,
                });
            }
            Err(_) => {
                cache.push(FitOut {
                    config,
                    preds: Vec::new(),
                    mae: f64::NAN,
                    fit_failed: true,
                });
            }
        }
    }
    Ok(cache)
}

pub fn cache_key(config: TrainConfig) -> TrainKey {
    (
        config.alpha.to_bits(),
        config.fit_intercept,
        config.solver_eps.to_bits(),
    )
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

fn predict_and_mae(data: &Matrix, beta: &[f64]) -> (Vec<f64>, f64) {
    let mut preds = Vec::with_capacity(data.rows);
    let mut mae = 0.0;
    for row in 0..data.rows {
        let mut pred = beta[0];
        let offset = row * data.cols;
        for col in 0..data.cols {
            pred += data.x[offset + col] * beta[col + 1];
        }
        mae += (pred - data.y[row]).abs();
        preds.push(pred);
    }
    (preds, mae / data.rows as f64)
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

fn read_plan() -> Result<Option<Vec<Param>>,Box<dyn std::error::Error>> {
    let path=match std::env::var_os("TERASWEEP_PLAN") {Some(p)=>p,None=>return Ok(None)};
    let text=std::fs::read_to_string(path)?;
    let mut rows=Vec::new();
    for (line_no,line) in text.lines().enumerate() {
        if line_no==0 {if line!="alpha,fit_intercept,solver_eps,threshold,fee_bps,slippage_bps,signal_rule" {return Err("invalid plan header".into());} continue;}
        let c:Vec<_>=line.split(',').collect();
        if c.len()!=7 {return Err("invalid plan row".into());}
        let signal_rule=match c[6] {"gt"=>results::SignalRule::Gt,"gte"=>results::SignalRule::Gte,"lt"=>results::SignalRule::Lt,"lte"=>results::SignalRule::Lte,"abs_gt"=>results::SignalRule::AbsGt,"abs_gte"=>results::SignalRule::AbsGte,_=>return Err("invalid plan rule".into())};
        rows.push(Param{id:rows.len(),train_ordinal:usize::MAX,threshold_ordinal:usize::MAX,train:TrainConfig{alpha:c[0].parse()?,fit_intercept:c[1].parse()?,solver_eps:c[2].parse()?},score:ScoreConfig{threshold:c[3].parse()?,fee_bps:c[4].parse()?,slippage_bps:c[5].parse()?,signal_rule}});
    }
    Ok(Some(rows))
}

#[cfg(all(test,target_arch="x86_64"))]
mod boundary_tests {
    use super::*;
    #[test]
    fn real_prediction_boundaries_and_masks() {
        let train=crate::read_matrix(include_bytes!("../data/train.bin")).unwrap();
        let test=crate::read_matrix(include_bytes!("../data/test.bin")).unwrap();
        let p=crate::read_prices(include_bytes!("../data/price.bin")).unwrap();
        let p=results::precompute_price_stats(&p);
        let stats=precompute_stats(&train);
        let cfg=[TrainConfig{alpha:1.0,fit_intercept:true,solver_eps:1e-12},TrainConfig{alpha:10.0,fit_intercept:false,solver_eps:1e-12}];
        let cache=precompute_train_cache(&stats,&test,&cfg).unwrap();
        let mut checked=0usize;
        for e in cache {
            assert!(!e.fit_failed);
            let mut thresholds=vec![0.0,-0.0];
            thresholds.extend(e.preds.iter().take(8));
            thresholds.push(e.preds.iter().copied().fold(f64::INFINITY,f64::min));
            thresholds.push(e.preds.iter().copied().fold(f64::NEG_INFINITY,f64::max));
            let nearby:Vec<_>=thresholds.iter().flat_map(|&x|[x,x.next_up(),x.next_down()]).collect();
            for n in [0,1,2,7,8,9,16,17,31,e.preds.len()] {
                for missing in [false,true] {
                    let mut prices=PriceStats{tradable:p.tradable[..n].to_vec(),r_entry:p.r_entry[..n].to_vec(),r_cont:p.r_cont[..n].to_vec()};
                    if missing {for (i,x) in prices.tradable.iter_mut().enumerate() {if i%11==0 {*x=false;}}}
                    let rules=[results::SignalRule::Gt,results::SignalRule::Gte,results::SignalRule::Lt,results::SignalRule::Lte,results::SignalRule::AbsGt,results::SignalRule::AbsGte];
                    let jobs:Vec<_>=nearby.iter().flat_map(|&t|rules.iter().map(move |&r|(t,r))).collect();
                    for used in 1..=8 {
                        for batch in jobs.chunks(used) {
                            for assembly in [false,true] {
                                let output=optimized::test_batch(&e.preds[..n],&prices,batch,assembly);
                                for (&(t,r),b) in batch.iter().zip(output) {assert!(optimized::same_base(&b,&results::compute_base_score(&e.preds[..n],t,r,&prices)));checked+=1;}
                            }
                        }
                    }
                }
            }
        }
        eprintln!("Boundary and mask full-BaseScore comparisons: {checked}");
    }
}
