use crate::model::{fit_predict, Model};
use crate::optimized::{self, Bounds, CachedFit, CachedTrial, Level};
use crate::results::{
    apply_costs, compute_backtest_stats, compute_base_score, precompute_price_stats, round_to,
    CommonSpace, Prices, ScoreConfig, ScoredRow, SignalRule, SweepSummary,
};
use crate::{read_matrix, read_prices, Matrix};

const RULES: [SignalRule; 6] = [
    SignalRule::Gt,
    SignalRule::Gte,
    SignalRule::Lt,
    SignalRule::Lte,
    SignalRule::AbsGt,
    SignalRule::AbsGte,
];

fn score(signal_rule: SignalRule, threshold: f64) -> ScoreConfig {
    ScoreConfig {
        threshold,
        signal_rule,
        fee_bps: 0.0,
        slippage_bps: 0.0,
    }
}

fn prices() -> Prices {
    Prices {
        open: vec![100.0, 100.0, 106.0, 105.0, 0.0, 105.0, 100.0, 105.0],
        close: vec![100.0, 110.0, 105.0, 100.0, 103.0, 105.0, 106.0, 100.0],
    }
}

fn common() -> CommonSpace {
    CommonSpace {
        seed: 42,
        signal_rules: RULES.to_vec(),
        threshold_signed: vec![0.0, -0.0, 1.0],
        threshold_abs: vec![0.0, 0.5, 1.0],
        fee_bps: vec![0.0, 17.0],
        slippage_bps: vec![0.0, 9.0],
        rolling_window_bars: 4,
        bar_minutes: 15.0,
    }
}

#[test]
fn native_models_match_reference_at_ten_decimals() {
    let train = Matrix {
        rows: 7,
        cols: 1,
        x: vec![-1.5, 0.25, 1.25, -0.5, 0.75, -0.25, 1.5],
        y: vec![0.5, -1.25, 1.0, 0.75, -1.5, 0.25, 1.5],
    };
    let test = Matrix {
        rows: 4,
        cols: 1,
        x: vec![-0.75, 1.75, -1.25, 0.0],
        y: vec![0.0; 4],
    };
    // Limen 6.0.1 with scikit-learn 1.7.2 Cholesky and NumPy 2.2.6 SVD.
    let cases = [
        (
            Model::Ridge {
                alpha: 0.75,
                fit_intercept: false,
                solver_eps: 1e-12,
            },
            [
                -0.084_677_419_354_838_72,
                0.197_580_645_161_290_34,
                -0.141_129_032_258_064_53,
                0.0,
            ],
        ),
        (
            Model::Ridge {
                alpha: 0.75,
                fit_intercept: true,
                solver_eps: 1e-12,
            },
            [
                0.099_759_615_384_615_38,
                0.304_086_538_461_538_55,
                0.058_894_230_769_230_74,
                0.161_057_692_307_692_32,
            ],
        ),
        (
            Model::DLinear {
                alpha: 0.75,
                lookback: 3,
                kernel: 3,
                solver_eps: 1e-12,
            },
            [
                -0.088_818_729_813_266_58,
                0.130_145_585_802_182_4,
                1.101_654_901_864_885_6,
                -0.840_855_019_759_589_4,
            ],
        ),
    ];
    for (model, expected) in cases {
        let actual = fit_predict(&model, &train, &test).unwrap().predictions;
        assert_eq!(actual.len(), expected.len());
        for (row, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let context = format!("{model:?} row {row}: {actual:.17} vs {expected:.17}");
            assert!((actual - expected).abs() <= 0.5e-10, "{context}");
            assert_eq!(round_to(actual, 10), round_to(expected, 10), "{context}");
            assert_eq!(actual > 0.0, expected > 0.0, "{context}");
        }
    }
}

#[test]
fn scalar_signal_boundaries_and_constant_shortcuts() {
    let prices = precompute_price_stats(&prices());
    let predictions = [
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for (rule, count) in RULES.into_iter().zip([2, 4, 2, 4, 4, 6]) {
        for threshold in [-0.0, 0.0] {
            let base = compute_base_score(&predictions, threshold, rule, &prices);
            assert_eq!(base.signal_count, count, "{}", rule.as_str());
            assert_eq!(
                Bounds::new(&predictions).constant(score(rule, threshold)),
                None
            );
        }
    }
    let finite = Bounds::new(&[-1.0, 1.0]);
    assert_eq!(finite.constant(score(SignalRule::Gt, 1.0)), Some(0));
    assert_eq!(finite.constant(score(SignalRule::Gte, 1.0)), None);
    assert_eq!(finite.constant(score(SignalRule::AbsGte, 1.0)), Some(1));
}

#[test]
fn factored_costs_match_direct_cash_flows_and_output_precision() {
    let p = precompute_price_stats(&prices());
    let predictions = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    let base = compute_base_score(&predictions, 0.0, SignalRule::Gt, &p);
    assert_eq!(
        (base.entry_count, base.exit_count, base.sharpe_count),
        (2, 2, 6)
    );
    assert_eq!(base.cat_count, [3, 1, 1, 1]);
    // Direct cash flows are 110/100, 105/110, 100/105, 1, 106/100, 1.
    // Apply costs on each entry and exit, then compound and compute sample Sharpe.
    for (fee, slip, net, sharpe, trade_median) in [
        (0.0, 0.0, 6.0, 0.19, 300.0),
        (17.0, 9.0, 4.9, 0.16, 246.549),
    ] {
        let config = ScoreConfig {
            fee_bps: fee,
            slippage_bps: slip,
            ..score(SignalRule::Gt, 0.0)
        };
        assert_eq!(
            apply_costs(&base, predictions.len(), config),
            (75.0, net, sharpe)
        );
        let stats = compute_backtest_stats(&predictions, config, &p, &common());
        assert_eq!(stats.trade_n, 2);
        assert_eq!(stats.trade_pnl_net_bps_p50, trade_median);
    }
    let flat = compute_base_score(&[0.0; 8], 0.0, SignalRule::Gt, &p);
    let (signal_rate, net_return, sharpe) = apply_costs(&flat, 8, score(SignalRule::Gt, 0.0));
    assert_eq!((signal_rate, net_return), (0.0, 0.0));
    assert!(sharpe.is_nan());
    assert_eq!(round_to(1.2345, 3), 1.235);
    assert_eq!(round_to(-1.2345, 3), -1.235);
}

fn winner(row: Option<ScoredRow>) -> (usize, [u64; 4]) {
    let row = row.unwrap();
    (
        row.id,
        [row.mae, row.signal_rate, row.net_return, row.sharpe].map(f64::to_bits),
    )
}

#[test]
fn compact_preserves_cache_keys_and_winners_across_blocks() {
    let p = precompute_price_stats(&prices());
    let common = common();
    let fits = [
        CachedFit {
            preds: &[1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.0, 1.0],
            mae: 0.0014,
            fit_failed: false,
        },
        CachedFit {
            preds: &[2.0; 8],
            mae: 0.0009,
            fit_failed: false,
        },
        CachedFit {
            preds: &[],
            mae: f64::NAN,
            fit_failed: true,
        },
    ];
    let trials = [
        (0, 0, SignalRule::Gt, 0.0, 0.0),
        (0, 0, SignalRule::Gt, 0.0, 17.0),
        (1, 0, SignalRule::Gt, 0.0, 0.0),
        (2, 0, SignalRule::Gt, 0.0, 0.0),
        (0, 1, SignalRule::Gt, -0.0, 0.0),
        (0, 1, SignalRule::Gte, -0.0, 0.0),
        (1, 0, SignalRule::Lt, 0.0, 0.0),
        (0, usize::MAX, SignalRule::Gt, 0.0, 0.0),
        (2, 0, SignalRule::Gte, 0.0, 0.0),
        (0, 2, SignalRule::AbsGt, 1.0, 0.0),
    ];
    let sample = |id: usize| {
        let (train_idx, threshold_ordinal, rule, threshold, fee_bps) = trials[id];
        CachedTrial {
            train_idx,
            threshold_ordinal,
            score: ScoreConfig {
                fee_bps,
                ..score(rule, threshold)
            },
        }
    };
    let mut scalar = SweepSummary::new();
    assert_eq!(
        optimized::sweep_cached(
            Level::Train,
            trials.len(),
            sample,
            &common,
            &fits,
            &p,
            &mut scalar
        )
        .unwrap(),
        0
    );
    assert_eq!(
        winner(scalar.best_mae()).0,
        0,
        "first rounded MAE wins a tie"
    );
    for level in [Level::Score, Level::Bounds, Level::Compact] {
        let mut actual = SweepSummary::new();
        let keys = if level == Level::Compact {
            optimized::compact::<4>(level, trials.len(), sample, &common, &fits, &p, &mut actual)
        } else {
            optimized::sweep_cached(level, trials.len(), sample, &common, &fits, &p, &mut actual)
        };
        assert_eq!(keys.unwrap(), 6, "{level:?}");
        assert_eq!((actual.rows(), actual.failed_rows()), (10, 2));
        assert_eq!(actual.signal_rule_counts(), [6, 2, 1, 0, 1, 0]);
        assert_eq!(
            winner(actual.best_return()),
            winner(scalar.best_return()),
            "{level:?}"
        );
        assert_eq!(
            winner(actual.best_sharpe()),
            winner(scalar.best_sharpe()),
            "{level:?}"
        );
        assert_eq!(
            winner(actual.best_mae()),
            winner(scalar.best_mae()),
            "{level:?}"
        );
    }
}

#[test]
fn summary_excludes_failed_fits_and_nonfinite_metrics() {
    let mut summary = SweepSummary::new();
    let mut row = ScoredRow {
        id: 0,
        signal_rule: SignalRule::Gt,
        fit_failed: true,
        mae: 0.0,
        signal_rate: 100.0,
        net_return: 999.0,
        sharpe: 999.0,
    };
    summary.record(row);
    row = ScoredRow {
        id: 1,
        fit_failed: false,
        mae: 1.0,
        net_return: f64::INFINITY,
        sharpe: f64::NAN,
        ..row
    };
    summary.record(row);
    assert!(summary.best_return().is_none());
    assert!(summary.best_sharpe().is_none());
    row = ScoredRow {
        id: 2,
        net_return: 2.5,
        sharpe: 0.5,
        ..row
    };
    summary.record(row);
    summary.record(ScoredRow { id: 3, ..row });
    assert_eq!((summary.rows(), summary.failed_rows()), (4, 1));
    assert_eq!(summary.best_mae().unwrap().id, 1);
    assert_eq!(summary.best_return().unwrap().id, 2);
    assert_eq!(summary.best_sharpe().unwrap().id, 2);
}

#[test]
fn fixture_decoders_reject_truncation_and_size_overflow() {
    let mut matrix = b"RIDGE001".to_vec();
    matrix.extend(1u64.to_le_bytes());
    matrix.extend(1u64.to_le_bytes());
    matrix.extend(2.0f64.to_le_bytes());
    assert!(read_matrix(&matrix).is_err());
    matrix.extend((-0.0f64).to_le_bytes());
    let parsed = read_matrix(&matrix).unwrap();
    assert_eq!(parsed.x, [2.0]);
    assert_eq!(parsed.y[0].to_bits(), (-0.0f64).to_bits());
    matrix[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(read_matrix(&matrix).is_err());

    let mut price = b"PRICE001".to_vec();
    price.extend((1u64 << 60).to_le_bytes());
    assert!(read_prices(&price).is_err());
    price[8..16].copy_from_slice(&1u64.to_le_bytes());
    price.extend(100.0f64.to_le_bytes());
    assert!(read_prices(&price).is_err());
    price.extend(105.0f64.to_le_bytes());
    let parsed = read_prices(&price).unwrap();
    assert_eq!(parsed.open, [100.0]);
    assert_eq!(parsed.close, [105.0]);
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn available_simd_kernels_match_scalar_bits() {
    use crate::results::PriceStats;

    let sample = [-1.0, -0.0, 0.0, 1.0, 0.5, -0.5, 0.25, -0.25];
    let predictions: Vec<_> = sample.into_iter().cycle().take(17).collect();
    let raw = Prices {
        open: (0..17)
            .map(|i| if i % 5 == 0 { 0.0 } else { 100.0 + i as f64 })
            .collect(),
        close: (0..17).map(|i| 102.0 + (i % 6) as f64).collect(),
    };
    let prices = precompute_price_stats(&raw);
    let jobs: Vec<_> = RULES
        .into_iter()
        .flat_map(|rule| {
            [-0.0f64, 0.0, 0.5f64.next_down(), 0.5, 0.5f64.next_up()]
                .into_iter()
                .map(move |threshold| (threshold, rule))
        })
        .collect();
    for assembly in [false, true] {
        if optimized::simd_available(assembly).is_err() {
            assert!(
                std::env::var_os("TERASWEEP_REQUIRE_SIMD").is_none(),
                "required SIMD kernel is unavailable"
            );
            eprintln!("Skipping unavailable SIMD kernel (assembly={assembly})");
            continue;
        }
        for n in [0, 1, 2, 7, 8, 9, 17] {
            let p = PriceStats {
                tradable: prices.tradable[..n].to_vec(),
                r_entry: prices.r_entry[..n].to_vec(),
                r_cont: prices.r_cont[..n].to_vec(),
            };
            for lanes in 1..=8 {
                for batch in jobs.chunks(lanes) {
                    let results = optimized::test_batch(&predictions[..n], &p, batch, assembly);
                    for (&(threshold, rule), result) in batch.iter().zip(results) {
                        let expected = compute_base_score(&predictions[..n], threshold, rule, &p);
                        assert!(
                            optimized::same_base(&result, &expected),
                            "assembly={assembly} rows={n} lanes={lanes} {} {threshold}",
                            rule.as_str()
                        );
                    }
                }
            }
        }
    }
}
