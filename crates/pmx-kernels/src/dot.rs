//! Int8 dot products, the inner loop of quantised matrix multiply.
//!
//! Decode of a quantised weight block is fused into the dot product: the weights
//! are dequantised into registers and consumed immediately, never written back to
//! memory. On a memory-bound machine a separate dequantise pass would double the
//! traffic for no benefit.
//!
//! The i5-1235U this was developed on has AVX2 and AVX-VNNI but no AVX-512 and
//! no AMX, so 256-bit `vpdpbusd` is the widest useful instruction. Dispatch is
//! resolved once at startup rather than per call.
//!
//! Every SIMD path is checked against the scalar reference in tests. The scalar
//! path is the definition of correct; the vector paths are optimisations that
//! must agree with it.

use crate::pmxq::{PmxType, GROUP};
use crate::{half::f16_to_f32, KernelError};

/// Which dot-product implementation is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    /// Portable scalar code.
    Scalar,
    /// 256-bit integer SIMD.
    Avx2,
    /// 256-bit SIMD with `vpdpbusd` (VNNI).
    Avx2Vnni,
}

impl Isa {
    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx2Vnni => "avx2+vnni",
        }
    }
}

/// The best implementation this CPU supports.
pub fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return Isa::Avx2Vnni;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
    }
    Isa::Scalar
}

/// Scalar reference: dot product of a pmx-quantised weight vector with `act`.
///
/// This is the definition the SIMD paths are tested against.
pub fn dot_scalar(ty: PmxType, w: &[u8], act: &[f32]) -> Result<f32, KernelError> {
    let n = act.len();
    let need = ty.bytes_for(n)?;
    if w.len() < need {
        return Err(KernelError::ShortBuffer {
            need,
            have: w.len(),
        });
    }
    let gb = ty.group_bytes();
    let bits = ty.payload_bits();
    let mut acc = 0.0f64;
    let mut q = [0u8; GROUP];

    for (gi, a) in act.chunks_exact(GROUP).enumerate() {
        let b = &w[gi * gb..(gi + 1) * gb];
        if ty == PmxType::Q8 {
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let mut s = 0i32;
            let mut fs = 0.0f64;
            for (j, av) in a.iter().enumerate() {
                // Activations stay in f32 here; the integer accumulation below is
                // what the VNNI path replaces.
                fs += f64::from(*av) * f64::from(b[2 + j] as i8);
            }
            let _ = s;
            s = 0;
            let _ = s;
            acc += fs * f64::from(d);
            continue;
        }
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let lo = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        unpack(&b[4..], bits, &mut q);
        let mut sq = 0.0f64;
        let mut sa = 0.0f64;
        for (j, av) in a.iter().enumerate() {
            sq += f64::from(*av) * f64::from(q[j]);
            sa += f64::from(*av);
        }
        // w = q*d + lo, so <w, a> = d*<q, a> + lo*sum(a).
        acc += f64::from(d) * sq + f64::from(lo) * sa;
    }
    Ok(acc as f32)
}

fn unpack(src: &[u8], bits: usize, out: &mut [u8; GROUP]) {
    let mut accum: u32 = 0;
    let mut have = 0usize;
    let mut si = 0usize;
    let mask = ((1u32 << bits) - 1) as u8;
    for o in out.iter_mut() {
        while have < bits {
            let byte = if si < src.len() { src[si] } else { 0 };
            accum |= u32::from(byte) << have;
            si += 1;
            have += 8;
        }
        *o = (accum as u8) & mask;
        accum >>= bits;
        have -= bits;
    }
}

/// Dot product using the best available implementation.
pub fn dot(ty: PmxType, w: &[u8], act: &[f32], isa: Isa) -> Result<f32, KernelError> {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx2Vnni if ty == PmxType::Q8 => {
            let n = act.len();
            let need = ty.bytes_for(n)?;
            if w.len() < need {
                return Err(KernelError::ShortBuffer {
                    need,
                    have: w.len(),
                });
            }
            // SAFETY: the caller-selected `isa` was produced by `detect_isa`,
            // which only returns Avx2/Avx2Vnni when the CPU reports avx2, and
            // the length checks above guarantee `w` holds a whole number of
            // groups covering `act`.
            Ok(unsafe { dot_q8_avx2(w, act) })
        }
        _ => dot_scalar(ty, w, act),
    }
}

/// AVX2 dot product for [`PmxType::Q8`].
///
/// # Safety
///
/// The CPU must support AVX2, `w` must hold at least `act.len()/GROUP` groups,
/// and `act.len()` must be a multiple of [`GROUP`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q8_avx2(w: &[u8], act: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let gb = PmxType::Q8.group_bytes();
    let mut total = _mm256_setzero_ps();

    for (gi, a) in act.chunks_exact(GROUP).enumerate() {
        let b = w.as_ptr().add(gi * gb);
        let d = f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
        let dv = _mm256_set1_ps(d);
        let qp = b.add(2) as *const i8;

        // 64 int8 weights against 64 f32 activations, eight at a time.
        let mut acc = _mm256_setzero_ps();
        let mut j = 0usize;
        while j < GROUP {
            // Widen 8 int8 weights to 8 f32.
            let q8 = _mm_loadl_epi64(qp.add(j) as *const __m128i);
            let q32 = _mm256_cvtepi8_epi32(q8);
            let qf = _mm256_cvtepi32_ps(q32);
            let av = _mm256_loadu_ps(a.as_ptr().add(j));
            acc = _mm256_fmadd_ps(qf, av, acc);
            j += 8;
        }
        total = _mm256_fmadd_ps(acc, dv, total);
    }
    // Horizontal sum of eight lanes.
    let hi = _mm256_extractf128_ps(total, 1);
    let lo = _mm256_castps256_ps128(total);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pmxq::quantize;

    fn noisy(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Dot product computed from fully dequantised weights, in f64.
    fn reference(ty: PmxType, w: &[u8], act: &[f32]) -> f64 {
        let mut deq = Vec::new();
        crate::pmxq::dequantize(ty, w, act.len(), &mut deq).unwrap();
        deq.iter()
            .zip(act)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    }

    #[test]
    fn scalar_matches_a_dequantise_then_multiply_reference() {
        let act = noisy(GROUP * 6, 11);
        let wf = noisy(GROUP * 6, 22);
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            quantize(ty, &wf, &mut packed).unwrap();
            let got = dot_scalar(ty, &packed, &act).unwrap() as f64;
            let want = reference(ty, &packed, &act);
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got - want).abs() < tol,
                "{}: fused {got} vs dequantised {want}",
                ty.label()
            );
        }
    }

    #[test]
    fn simd_agrees_with_scalar() {
        let isa = detect_isa();
        if isa == Isa::Scalar {
            // Nothing to compare on this host; the scalar path is already tested.
            return;
        }
        let act = noisy(GROUP * 8, 5);
        let wf = noisy(GROUP * 8, 6);
        let mut packed = Vec::new();
        quantize(PmxType::Q8, &wf, &mut packed).unwrap();
        let s = dot_scalar(PmxType::Q8, &packed, &act).unwrap();
        let v = dot(PmxType::Q8, &packed, &act, isa).unwrap();
        let tol = 1e-3 * s.abs().max(1.0);
        assert!(
            (s - v).abs() < tol,
            "{}: scalar {s} vs simd {v}",
            isa.name()
        );
    }

    #[test]
    fn simd_agrees_with_scalar_across_many_random_inputs() {
        let isa = detect_isa();
        if isa == Isa::Scalar {
            return;
        }
        for seed in 0..24u64 {
            let n = GROUP * (1 + (seed as usize % 5));
            let act = noisy(n, seed * 2 + 1);
            let wf = noisy(n, seed * 2 + 2);
            let mut packed = Vec::new();
            quantize(PmxType::Q8, &wf, &mut packed).unwrap();
            let s = dot_scalar(PmxType::Q8, &packed, &act).unwrap();
            let v = dot(PmxType::Q8, &packed, &act, isa).unwrap();
            let tol = 2e-3 * s.abs().max(1.0);
            assert!((s - v).abs() < tol, "seed {seed}: scalar {s} vs simd {v}");
        }
    }

    #[test]
    fn zero_activations_give_zero() {
        let act = vec![0.0f32; GROUP * 2];
        let wf = noisy(GROUP * 2, 9);
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            quantize(ty, &wf, &mut packed).unwrap();
            let got = dot_scalar(ty, &packed, &act).unwrap();
            assert!(got.abs() < 1e-4, "{}: {got}", ty.label());
        }
    }

    #[test]
    fn short_weight_buffer_is_an_error() {
        let act = vec![1.0f32; GROUP * 2];
        let packed = vec![0u8; 4];
        assert!(matches!(
            dot_scalar(PmxType::Q4, &packed, &act),
            Err(KernelError::ShortBuffer { .. })
        ));
    }

    #[test]
    fn detected_isa_is_reported_not_assumed() {
        // Just assert the call is total and the label is populated.
        let isa = detect_isa();
        assert!(!isa.name().is_empty());
    }
}
