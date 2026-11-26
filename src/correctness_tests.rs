use crate::{fit_predict, read_matrix, sweep, Matrix, Model};

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
    // Direct scikit-learn 1.7.2 Cholesky and NumPy 2.2.6 SVD references.
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
            assert_eq!(
                format!("{actual:.10}"),
                format!("{expected:.10}"),
                "{context}"
            );
        }
    }
}

#[test]
fn sweep_matches_single_fits_and_counts_distinct_configurations() {
    let train = Matrix {
        rows: 6,
        cols: 1,
        x: vec![0.0, 1.0, 2.0, 4.0, 5.0, 7.0],
        y: vec![1.0, 2.5, 4.0, 6.0, 8.0, 11.0],
    };
    let validation = Matrix {
        rows: 3,
        cols: 1,
        x: vec![3.0, 6.0, 8.0],
        y: vec![5.5, 9.5, 13.0],
    };
    let ridge = |alpha| Model::Ridge {
        alpha,
        fit_intercept: true,
        solver_eps: 0.0,
    };
    let dlinear = |alpha| Model::DLinear {
        alpha,
        lookback: 3,
        kernel: 3,
        solver_eps: 0.0,
    };
    let models = [
        ridge(16.0),
        dlinear(2.0),
        ridge(0.5),
        dlinear(4.0),
        ridge(0.5),
    ];
    let (trial, (mae, predictions)) = models
        .iter()
        .enumerate()
        .map(|(trial, model)| {
            let predictions = fit_predict(model, &train, &validation).unwrap().predictions;
            let mae = predictions
                .iter()
                .zip(&validation.y)
                .map(|(p, y)| (p - y).abs())
                .sum::<f64>()
                / 3.0;
            (trial, (mae, predictions))
        })
        .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
        .unwrap();

    let actual = sweep(&models, &train, &validation).unwrap();
    assert_eq!(actual.trials, models.len());
    assert_eq!(actual.unique_fits, 4);
    assert_eq!(actual.best.trial, trial);
    assert_eq!(actual.best.model, models[trial]);
    assert_eq!(actual.best.mae.to_bits(), mae.to_bits());
    assert_eq!(
        actual
            .best
            .predictions
            .iter()
            .map(|p| p.to_bits())
            .collect::<Vec<_>>(),
        predictions.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
    );
}

#[test]
fn equal_scores_choose_first_plan_entry() {
    let data = Matrix {
        rows: 2,
        cols: 1,
        x: vec![0.0, 0.0],
        y: vec![2.0, 2.0],
    };
    let models = [0.1, 1.0].map(|alpha| Model::Ridge {
        alpha,
        fit_intercept: true,
        solver_eps: 0.0,
    });
    let actual = sweep(&models, &data, &data).unwrap();
    assert_eq!(actual.unique_fits, 2);
    assert_eq!(actual.best.trial, 0);
    assert_eq!(actual.best.mae, 0.0);
}

#[test]
fn invalid_sweeps_fail_before_reporting_a_winner() {
    let data = Matrix {
        rows: 1,
        cols: 1,
        x: vec![1.0],
        y: vec![2.0],
    };
    let ridge = |alpha| Model::Ridge {
        alpha,
        fit_intercept: false,
        solver_eps: 0.0,
    };
    let invalid = Matrix {
        rows: 1,
        cols: 1,
        x: vec![1.0],
        y: vec![f64::NAN],
    };
    assert!(sweep(&[], &data, &data).is_err());
    assert!(sweep(&[ridge(1.0)], &data, &invalid).is_err());
    assert!(sweep(&[ridge(1.0), ridge(f64::INFINITY)], &data, &data)
        .unwrap_err()
        .to_string()
        .contains("Trial 1"));
    let too_large = Model::DLinear {
        alpha: 1.0,
        lookback: usize::MAX,
        kernel: 3,
        solver_eps: 0.0,
    };
    assert!(sweep(&[too_large], &data, &data).is_err());
}

#[test]
fn matrix_decoder_checks_dimensions_and_exact_length() {
    let mut bytes = b"RIDGE001".to_vec();
    for _ in 0..2 {
        bytes.extend_from_slice(&1u64.to_le_bytes());
    }
    bytes.extend_from_slice(&2f64.to_le_bytes());
    bytes.extend_from_slice(&3f64.to_le_bytes());
    let data = read_matrix(&bytes).unwrap();
    assert_eq!(data.x, [2.0]);
    assert_eq!(data.y, [3.0]);
    for length in 0..bytes.len() {
        assert!(read_matrix(&bytes[..length]).is_err());
    }
    bytes.push(0);
    assert!(read_matrix(&bytes).is_err());
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(read_matrix(&bytes).is_err());
}
