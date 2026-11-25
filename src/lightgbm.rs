use std::error::Error;

use crate::{lightgbm_api::LightGbm, Matrix};

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub num_trees: usize,
    pub num_leaves: usize,
    pub max_bins: usize,
    pub lambda_l2: f64,
    pub min_data_in_leaf: usize,
}

pub(crate) struct Backend {
    reference: LightGbm,
    labels: Vec<f32>,
}

impl Backend {
    pub(crate) fn new(train: &Matrix) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            reference: LightGbm::load()?,
            labels: labels(train),
        })
    }

    pub(crate) fn fit_predict(
        &self,
        train: &Matrix,
        test: &Matrix,
        config: TrainConfig,
        seed: u32,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        self.reference
            .fit_predict(train, &self.labels, test, config, seed)
    }
}

pub(crate) fn fit_one(
    train: &Matrix,
    test: &Matrix,
    config: TrainConfig,
    seed: u32,
) -> Result<crate::FitPrediction, Box<dyn Error>> {
    let reference = LightGbm::load()?;
    crate::model::measure(|| reference.fit_predict(train, &labels(train), test, config, seed))
}

fn labels(train: &Matrix) -> Vec<f32> {
    train.y.iter().map(|&value| value as f32).collect()
}
