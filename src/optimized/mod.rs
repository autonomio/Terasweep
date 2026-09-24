use std::collections::HashMap;
use std::time::Instant;
use crate::results::{self, BaseScore, PriceStats, CommonSpace, ScoreConfig, ScoredRow, SignalRule, SweepSummary};
use crate::ridge::{FitOut, Param, TrainKey};

#[cfg(target_arch = "x86_64")]
mod simd;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level { Naive, Stats, Train, Score, Bounds, Compact, Intrinsics, Assembly }
impl Level {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let level = match std::env::var("TERASWEEP_LEVEL").unwrap_or_else(|_| "score".into()).as_str() {
            "naive" => Self::Naive, "stats" => Self::Stats, "train" => Self::Train,
            "score" => Self::Score, "bounds" => Self::Bounds, "compact" => Self::Compact,
            "intrinsics" => Self::Intrinsics, "assembly" => Self::Assembly,
            _ => return Err("TERASWEEP_LEVEL must be naive/stats/train/score/bounds/compact/intrinsics/assembly".into()),
        };
        if matches!(level, Self::Intrinsics | Self::Assembly) {
            #[cfg(target_arch="x86_64")]
            simd::check(level == Self::Assembly)?;
            #[cfg(not(target_arch="x86_64"))]
            return Err("AVX-512 levels require an x86-64 machine".into());
        }
        Ok(level)
    }
}

#[derive(Clone, Copy)]
pub struct Bounds { lo: f64, hi: f64, abs_lo: f64, abs_hi: f64, finite: bool }
impl Bounds {
    pub fn new(preds: &[f64]) -> Self {
        let mut b = Self { lo: f64::INFINITY, hi: f64::NEG_INFINITY, abs_lo: f64::INFINITY, abs_hi: 0.0, finite: !preds.is_empty() };
        for &x in preds {
            b.finite &= x.is_finite(); b.lo = b.lo.min(x); b.hi = b.hi.max(x);
            b.abs_lo = b.abs_lo.min(x.abs()); b.abs_hi = b.abs_hi.max(x.abs());
        }
        b
    }
    pub fn constant(&self, s: ScoreConfig) -> Option<usize> {
        if !self.finite { return None; }
        let t = s.threshold;
        let (off,on) = match s.signal_rule {
            SignalRule::Gt => (self.hi <= t, self.lo > t),
            SignalRule::Gte => (self.hi < t, self.lo >= t),
            SignalRule::Lt => (self.lo >= t, self.hi < t),
            SignalRule::Lte => (self.lo > t, self.hi <= t),
            SignalRule::AbsGt => (self.abs_hi <= t, self.abs_lo > t),
            SignalRule::AbsGte => (self.abs_hi < t, self.abs_lo >= t),
        };
        if off { Some(0) } else if on { Some(1) } else { None }
    }
}

pub fn same_base(a: &BaseScore, b: &BaseScore) -> bool {
    a.signal_count == b.signal_count && a.entry_count == b.entry_count && a.exit_count == b.exit_count
    && a.sharpe_count == b.sharpe_count && a.eq_gross.to_bits() == b.eq_gross.to_bits()
    && a.cat_count == b.cat_count
    && a.cat_sum_a.iter().zip(b.cat_sum_a).all(|(x,y)| x.to_bits()==y.to_bits())
    && a.cat_sum_a2.iter().zip(b.cat_sum_a2).all(|(x,y)| x.to_bits()==y.to_bits())
}

pub struct OnlineBounds {
    ranges: Vec<Option<Bounds>>, constants: [Option<BaseScore>;2],
    pub counts: [usize;3], check: bool,
}
impl OnlineBounds {
    pub fn new(n: usize) -> Self { Self { ranges: vec![None;n], constants: [None;2], counts: [0;3], check: std::env::var_os("TERASWEEP_CHECK_BASE").is_some() } }
    pub fn base(&mut self, idx: usize, preds: &[f64], s: ScoreConfig, p: &PriceStats) -> BaseScore {
        let b = *self.ranges[idx].get_or_insert_with(|| Bounds::new(preds));
        let which = b.constant(s);
        self.counts[which.unwrap_or(2)] += 1;
        let base = if let Some(k) = which {
            *self.constants[k].get_or_insert_with(|| results::compute_base_score(preds,s.threshold,s.signal_rule,p))
        } else { results::compute_base_score(preds,s.threshold,s.signal_rule,p) };
        if self.check {
            assert!(same_base(&base,&results::compute_base_score(preds,s.threshold,s.signal_rule,p)), "bounds mismatch for training key {idx}");
        }
        base
    }
}

// Dense categorical addresses are bounded; large configured spaces retain a
// sparse map rather than allocating in proportion to a huge Cartesian product.
struct Slots { dense: Option<Vec<u32>>, sparse: HashMap<usize,u32>, len: usize }
impl Slots {
    fn new(possible: usize, requested: usize) -> Self {
        let dense = if possible <= 64*1024*1024/std::mem::size_of::<u32>() {Some(vec![u32::MAX;possible])} else {None};
        let sparse=if dense.is_some() {HashMap::new()} else {HashMap::with_capacity(requested.min(possible).min(1<<20))};
        Self {dense,sparse,len:0}
    }
    fn get(&self,key:usize)->Option<u32> {
        match &self.dense {Some(v)=>{let n=v[key];if n==u32::MAX {None} else {Some(n)}},None=>self.sparse.get(&key).copied()}
    }
    fn insert(&mut self,key:usize,value:u32) {
        match &mut self.dense {Some(v)=>v[key]=value,None=>{self.sparse.insert(key,value);}}
        self.len+=1;
    }
    fn len(&self)->usize {self.len}
}

#[derive(Clone, Copy)]
struct RowRef { train: u32, slot: u32 }
struct Job { train: usize, score: ScoreConfig, slot: usize }

pub(crate) fn compact(
    level: Level, runs: usize, sample: impl Fn(usize)->Param,
    common: &CommonSpace, ordinal_to_cache: &[usize], train_index: &HashMap<TrainKey,usize>, cache: &[FitOut], p: &PriceStats,
    summary: &mut SweepSummary,
) -> Result<usize,Box<dyn std::error::Error>> {
    let start=Instant::now();
    let validate=std::env::var_os("TERASWEEP_CHECK_BASE").is_some();
    let mut threshold_ids:HashMap<(usize,u64),usize>=HashMap::new();
    let mut threshold_grid:[Vec<usize>;6]=std::array::from_fn(|_|Vec::new());
    for rule in 0..6 {
        let ts=if rule>=4 {&common.threshold_abs} else {&common.threshold_signed};
        for &t in ts {
            let n=threshold_ids.len();
            threshold_grid[rule].push(*threshold_ids.entry((rule,t.to_bits())).or_insert(n));
        }
    }
    let width=threshold_ids.len();
    let possible=cache.len().checked_mul(width).ok_or("score-key address space overflow")?;
    let mut table=Slots::new(possible,runs);
    const BLOCK_ROWS: usize=1<<20;
    let mut rows=Vec::with_capacity(runs.min(BLOCK_ROWS));
    let mut ranges=vec![None;cache.len()];
    let mut constants=[None;2];
    let mut jobs=Vec::new();
    let mut counts=[0usize;3];
    let mut bases:Vec<Option<BaseScore>>=Vec::new();
    let mut plan_s=start.elapsed().as_secs_f64();
    let mut score_s=0.0;
    let mut costs_s=0.0;
    for block_start in (0..runs).step_by(BLOCK_ROWS) {
        rows.clear(); jobs.clear();
        let start=Instant::now();
        let block_end=block_start.saturating_add(BLOCK_ROWS).min(runs);
    for id in block_start..block_end {
        let param=sample(id);
        let idx=if param.train_ordinal==usize::MAX {train_index[&crate::ridge::cache_key(param.train)]} else {ordinal_to_cache[param.train_ordinal]};
        let e=&cache[idx];
        if e.fit_failed { rows.push(RowRef{train:idx as u32,slot:u32::MAX}); continue; }
        let rule=param.score.signal_rule.index();
        let threshold=if param.threshold_ordinal==usize::MAX {
            *threshold_ids.get(&(rule,param.score.threshold.to_bits())).ok_or("explicit plan threshold is outside configured grid")?
        } else {threshold_grid[rule][param.threshold_ordinal]};
        let key=idx*width+threshold;
        let slot=if let Some(slot)=table.get(key) {slot} else {
            let slot = {
            let range=*ranges[idx].get_or_insert_with(|| Bounds::new(&e.preds));
            if let Some(k)=range.constant(param.score) {
                counts[k]+=1;
                let b=*constants[k].get_or_insert_with(|| results::compute_base_score(&e.preds,param.score.threshold,param.score.signal_rule,p));
                if validate { assert!(same_base(&b,&results::compute_base_score(&e.preds,param.score.threshold,param.score.signal_rule,p)),"constant mismatch"); }
                k as u32
            } else {
                counts[2]+=1;
                let slot=bases.len();
                bases.push(None);
                jobs.push(Job{train:idx,score:param.score,slot});
                (slot+2).try_into().expect("more than u32::MAX base scores")
            }
            };
            table.insert(key,slot);
            slot
        };
        rows.push(RowRef{train:idx as u32,slot});
    }
    plan_s+=start.elapsed().as_secs_f64();
    let start=Instant::now();
    let mut order: Vec<usize>=(0..jobs.len()).collect();
    order.sort_unstable_by_key(|&j| jobs[j].train);
    let mut at=0;
    while at<order.len() {
        let idx=jobs[order[at]].train;
        let mut end=at+1;
        while end<order.len() && end<at+8 && jobs[order[end]].train==idx {end+=1;}
        let pred=&cache[idx].preds;
        if level==Level::Compact || !ranges[idx].unwrap().finite {
            for &j in &order[at..end] {let job=&jobs[j]; bases[job.slot]=Some(results::compute_base_score(pred,job.score.threshold,job.score.signal_rule,p));}
        } else {
            #[cfg(target_arch="x86_64")]
            {
                let rules:Vec<_>=order[at..end].iter().map(|&j| (jobs[j].score.threshold,jobs[j].score.signal_rule)).collect();
                let scored=simd::batch(pred,p,&rules,level==Level::Assembly);
                for (&j,b) in order[at..end].iter().zip(scored) {
                    let job=&jobs[j];
                    if validate {assert!(same_base(&b,&results::compute_base_score(pred,job.score.threshold,job.score.signal_rule,p)),"SIMD mismatch at base slot {}",job.slot);}
                    bases[job.slot]=Some(b);
                }
            }
            #[cfg(not(target_arch="x86_64"))]
            unreachable!();
        }
        at=end;
    }
    score_s+=start.elapsed().as_secs_f64();
    let start=Instant::now();
    for (offset,row) in rows.iter().enumerate() {
        let id=block_start+offset;
        let param=sample(id);
        let e=&cache[row.train as usize];
        let scores=if e.fit_failed {(f64::NAN,f64::NAN,f64::NAN)} else {
            let b=if row.slot<2 {constants[row.slot as usize].as_ref().unwrap()} else {bases[row.slot as usize-2].as_ref().unwrap()};
            results::apply_costs(b,e.preds.len(),param.score)
        };
        summary.record(ScoredRow {id,signal_rule:param.score.signal_rule,fit_failed:e.fit_failed,mae:results::round_to(e.mae,3),signal_rate:scores.0,net_return:scores.1,sharpe:scores.2});
    }
    costs_s+=start.elapsed().as_secs_f64();
    }
    eprintln!("PIPELINE plan={plan_s:.9} mixed_score={score_s:.9} costs_reduce={costs_s:.9} unique={} constant_mixed={counts:?} validate={validate} row_buffer_capacity={}",table.len(),rows.capacity());
    // Keep exactly the original logical-cache cardinality, despite shared payloads.
    Ok(table.len())
}

#[cfg(all(test,target_arch="x86_64"))]
pub fn test_batch(preds:&[f64],p:&PriceStats,rules:&[(f64,SignalRule)],assembly:bool)->Vec<BaseScore>{
    simd::check(assembly).unwrap();
    simd::batch(preds,p,rules,assembly)
}
