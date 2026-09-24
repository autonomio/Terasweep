use crate::results::{BaseScore, PriceStats, SignalRule};
use std::arch::x86_64::*;

#[repr(C)]
pub struct PriceView {
    n: usize,
    tradable: *const u8,
    entry: *const f64,
    cont: *const f64,
}
#[repr(C)]
pub struct Packed8 {
    threshold: [f64; 8],
    sign_xor: [u64; 8],
    abs_and: [u64; 8],
    inclusive: u64,
}
#[repr(C)]
pub struct Raw8 {
    signal: [u64; 8],
    equity: [f64; 8],
    count: [[u64; 8]; 4],
    sum: [[f64; 8]; 4],
    square: [[f64; 8]; 4],
    sharpe: u64,
}
const _: () = assert!(std::mem::offset_of!(PriceView, cont) == 24);
const _: () = assert!(std::mem::offset_of!(Packed8, inclusive) == 192);
const _: () = assert!(std::mem::offset_of!(Raw8, sharpe) == 896);

type Kernel = unsafe extern "C" fn(*const f64, *const PriceView, *const Packed8, *mut Raw8);

// The authored assembly is linked directly into this Rust executable. No C code.
core::arch::global_asm!(include_str!("score8_gpr.S"), options(raw));
unsafe extern "C" {
    fn score8_asm_gpr(pred: *const f64, pz: *const PriceView, rules: *const Packed8, out: *mut Raw8);
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx512f")]
pub unsafe extern "C" fn score8_rust_intrinsics(
    pred: *const f64, pz: *const PriceView, rules: *const Packed8, out: *mut Raw8,
) {
    // SAFETY: the checked caller owns n-element input arrays, full Packed8/Raw8,
    // and dispatches here only after detecting AVX-512F at runtime.
    unsafe {
        let p = &*pz;
        let r = &*rules;
        let out = &mut *out;
        let zero = _mm512_setzero_pd();
        let one = _mm512_set1_pd(1.0);
        let oi = _mm512_set1_epi64(1);
        let t = _mm512_loadu_pd(r.threshold.as_ptr());
        let sx = _mm512_loadu_si512(r.sign_xor.as_ptr().cast());
        let ab = _mm512_loadu_si512(r.abs_and.as_ptr().cast());
        let inc = r.inclusive as __mmask8;
        let mut eq = one;
        let mut sums = [zero; 4];
        let mut squares = [zero; 4];
        let mut sigcnt = _mm512_setzero_si512();
        let mut counts = [sigcnt; 4];
        let mut last: __mmask8 = 0;
        let mut prev: __mmask8 = 0;
        let mut sharpe = 0;
        for i in 0..p.n {
            let pv = _mm512_castpd_si512(_mm512_set1_pd(*pred.add(i)));
            let pv = _mm512_castsi512_pd(_mm512_and_si512(_mm512_xor_si512(pv, sx), ab));
            let sig = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(pv, t)
                | _mm512_mask_cmp_pd_mask::<_CMP_EQ_OQ>(inc, pv, t);
            sigcnt = _mm512_mask_add_epi64(sigcnt, sig, sigcnt, oi);
            let tradable = *p.tradable.add(i) != 0;
            let pos = if i > 0 && tradable { last } else { 0 };
            let next = if i + 1 < p.n && *p.tradable.add(i + 1) != 0 { sig } else { 0 };
            let en = !prev & pos;
            let co = prev & pos;
            let ex = !next & pos;
            let gross = _mm512_mask_add_pd(zero, en, zero, _mm512_set1_pd(*p.entry.add(i)));
            let gross = _mm512_mask_add_pd(gross, co, gross, _mm512_set1_pd(*p.cont.add(i)));
            let a = _mm512_add_pd(one, gross);
            eq = _mm512_mul_pd(eq, a);
            if i > 0 && tradable {
                sharpe += 1;
                let aa = _mm512_mul_pd(a, a);
                let masks = [!(en | ex), !ex & en, !en & ex, en & ex];
                for c in 0..4 {
                    counts[c] = _mm512_mask_add_epi64(counts[c], masks[c], counts[c], oi);
                    sums[c] = _mm512_mask_add_pd(sums[c], masks[c], sums[c], a);
                    squares[c] = _mm512_mask_add_pd(squares[c], masks[c], squares[c], aa);
                }
            }
            prev = pos;
            last = sig;
        }
        _mm512_storeu_si512(out.signal.as_mut_ptr().cast(), sigcnt);
        _mm512_storeu_pd(out.equity.as_mut_ptr(), eq);
        for c in 0..4 {
            _mm512_storeu_si512(out.count[c].as_mut_ptr().cast(), counts[c]);
            _mm512_storeu_pd(out.sum[c].as_mut_ptr(), sums[c]);
            _mm512_storeu_pd(out.square[c].as_mut_ptr(), squares[c]);
        }
        out.sharpe = sharpe;
    }
}

pub fn check(assembly: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !is_x86_feature_detected!("avx512f") || (assembly && !is_x86_feature_detected!("bmi1")) {
        return Err("selected kernel requires AVX-512F (and BMI1 for assembly)".into());
    }
    Ok(())
}

pub fn batch(preds: &[f64], p: &PriceStats, rules: &[(f64, SignalRule)], assembly: bool) -> Vec<BaseScore> {
    assert!(!rules.is_empty() && rules.len() <= 8);
    assert_eq!(preds.len(), p.tradable.len());
    assert_eq!(preds.len(), p.r_entry.len());
    assert_eq!(preds.len(), p.r_cont.len());
    let mut packed = Packed8 { threshold: [0.0; 8], sign_xor: [0; 8], abs_and: [u64::MAX; 8], inclusive: 0 };
    for lane in 0..8 {
        let (threshold, rule) = rules[lane.min(rules.len() - 1)];
        let neg = matches!(rule, SignalRule::Lt | SignalRule::Lte);
        packed.threshold[lane] = if neg { -threshold } else { threshold };
        packed.sign_xor[lane] = if neg { 1u64 << 63 } else { 0 };
        packed.abs_and[lane] = if rule.is_abs() { u64::MAX >> 1 } else { u64::MAX };
        if rule.index() & 1 != 0 { packed.inclusive |= 1 << lane; }
    }
    let view = PriceView { n: preds.len(), tradable: p.tradable.as_ptr().cast(), entry: p.r_entry.as_ptr(), cont: p.r_cont.as_ptr() };
    let mut raw = std::mem::MaybeUninit::<Raw8>::uninit();
    let kernel: Kernel = if assembly { score8_asm_gpr } else { score8_rust_intrinsics };
    // SAFETY: sizes are asserted, pointers are live, and run dispatch checks ISA.
    unsafe { kernel(preds.as_ptr(), &view, &packed, raw.as_mut_ptr()); }
    let raw = unsafe { raw.assume_init() };
    (0..rules.len()).map(|l| {
        let cat_count = std::array::from_fn(|c| raw.count[c][l] as usize);
        BaseScore {
            signal_count: raw.signal[l] as usize,
            eq_gross: raw.equity[l],
            entry_count: (cat_count[1] + cat_count[3]) as i32,
            exit_count: (cat_count[2] + cat_count[3]) as i32,
            sharpe_count: raw.sharpe as usize,
            cat_count,
            cat_sum_a: std::array::from_fn(|c| raw.sum[c][l]),
            cat_sum_a2: std::array::from_fn(|c| raw.square[c][l]),
        }
    }).collect()
}
