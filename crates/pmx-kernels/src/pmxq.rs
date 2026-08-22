// SPDX-License-Identifier: GPL-2.0-or-later
//! potatomaxx native block formats.
//!
//! # Why not just reuse the k-quants
//!
//! Per-expert mixed precision cannot be expressed in GGUF: a tensor carries one
//! `ggml_type`, and a MoE layer's experts share a tensor. Assigning expert 7
//! three bits and expert 8 five bits therefore requires a store we define
//! ourselves.
//!
//! Given that, these formats are deliberately simple — an asymmetric
//! scale-and-offset per group of 64 weights — rather than a reimplementation of
//! the k-quant super-block machinery. A quantiser whose correctness is obvious
//! and round-trip-tested is worth more here than one that squeezes out the last
//! few hundredths of a bit, because it is the *allocation* of bits across
//! experts that this project is testing, not the quantiser.
//!
//! Group size 64 follows the same reasoning as Colibri's `gs64`: small enough to
//! track local dynamic range, large enough that the two f16 parameters cost
//! only 0.25 bits per weight.
//!
//! | format | bits | group | bytes/group | bits/weight |
//! |--------|------|-------|-------------|-------------|
//! | Q8     | 8    | 64    | 66          | 8.25        |
//! | Q5     | 5    | 64    | 44          | 5.50        |
//! | Q4     | 4    | 64    | 36          | 4.50        |
//! | Q3     | 3    | 64    | 28          | 3.50        |
//! | Q2     | 2    | 64    | 20          | 2.50        |

use crate::half::{f16_to_f32, f32_to_f16};
use crate::KernelError;

/// Weights per quantisation group.
pub const GROUP: usize = 64;

/// A potatomaxx block format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PmxType {
    /// 8-bit, symmetric (no offset).
    Q8,
    /// 5-bit, asymmetric.
    Q5,
    /// 4-bit, asymmetric.
    Q4,
    /// 3-bit, asymmetric.
    Q3,
    /// 2-bit, asymmetric.
    Q2,
}

impl PmxType {
    /// Every format, most precise first.
    pub const ALL: [PmxType; 5] = [
        PmxType::Q8,
        PmxType::Q5,
        PmxType::Q4,
        PmxType::Q3,
        PmxType::Q2,
    ];

    /// Bits per weight in the packed payload, excluding group parameters.
    pub fn payload_bits(self) -> usize {
        match self {
            PmxType::Q8 => 8,
            PmxType::Q5 => 5,
            PmxType::Q4 => 4,
            PmxType::Q3 => 3,
            PmxType::Q2 => 2,
        }
    }

    /// Bytes occupied by one group of [`GROUP`] weights.
    pub fn group_bytes(self) -> usize {
        let payload = GROUP * self.payload_bits() / 8;
        // Q8 stores a scale only; the others store scale and offset.
        let params = if self == PmxType::Q8 { 2 } else { 4 };
        payload + params
    }

    /// Effective bits per weight, group parameters included.
    pub fn bits_per_weight(self) -> f64 {
        (self.group_bytes() * 8) as f64 / GROUP as f64
    }

    /// Short label used in stores and reports.
    pub fn label(self) -> &'static str {
        match self {
            PmxType::Q8 => "pmxq8",
            PmxType::Q5 => "pmxq5",
            PmxType::Q4 => "pmxq4",
            PmxType::Q3 => "pmxq3",
            PmxType::Q2 => "pmxq2",
        }
    }

    /// Wire tag stored in a potatomaxx store index.
    pub fn tag(self) -> u8 {
        match self {
            PmxType::Q8 => 8,
            PmxType::Q5 => 5,
            PmxType::Q4 => 4,
            PmxType::Q3 => 3,
            PmxType::Q2 => 2,
        }
    }

    /// Recover a format from its wire tag.
    pub fn from_tag(t: u8) -> Option<PmxType> {
        Some(match t {
            8 => PmxType::Q8,
            5 => PmxType::Q5,
            4 => PmxType::Q4,
            3 => PmxType::Q3,
            2 => PmxType::Q2,
            _ => return None,
        })
    }

    /// Bytes needed to store `n` weights, which must be a multiple of [`GROUP`].
    pub fn bytes_for(self, n: usize) -> Result<usize, KernelError> {
        if n % GROUP != 0 {
            return Err(KernelError::PartialBlock {
                n_elements: n,
                block_elems: GROUP,
            });
        }
        Ok(n / GROUP * self.group_bytes())
    }
}

/// Pack `bits`-wide unsigned values, least significant bit first.
fn pack_bits(vals: &[u8], bits: usize, out: &mut Vec<u8>) {
    let mut acc: u32 = 0;
    let mut have = 0usize;
    for &v in vals {
        acc |= u32::from(v) << have;
        have += bits;
        while have >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push((acc & 0xFF) as u8);
    }
}

/// Inverse of [`pack_bits`].
fn unpack_bits(src: &[u8], bits: usize, n: usize, out: &mut [u8]) {
    let mut acc: u32 = 0;
    let mut have = 0usize;
    let mut si = 0usize;
    let mask = ((1u32 << bits) - 1) as u8;
    for o in out.iter_mut().take(n) {
        while have < bits {
            let byte = if si < src.len() { src[si] } else { 0 };
            acc |= u32::from(byte) << have;
            si += 1;
            have += 8;
        }
        *o = (acc as u8) & mask;
        acc >>= bits;
        have -= bits;
    }
}

/// Quantise `x` (a multiple of [`GROUP`] values) into `out`.
pub fn quantize(ty: PmxType, x: &[f32], out: &mut Vec<u8>) -> Result<(), KernelError> {
    if x.len() % GROUP != 0 {
        return Err(KernelError::PartialBlock {
            n_elements: x.len(),
            block_elems: GROUP,
        });
    }
    out.clear();
    out.reserve(ty.bytes_for(x.len())?);
    let levels = (1u32 << ty.payload_bits()) - 1;
    let mut q = vec![0u8; GROUP];

    for g in x.chunks_exact(GROUP) {
        if ty == PmxType::Q8 {
            // Symmetric: one scale, quants are signed.
            let amax = g.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for &v in g {
                let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                out.push(qi as u8);
            }
            continue;
        }
        // Asymmetric: scale and offset, quants unsigned in 0..=levels.
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in g {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        let d = (hi - lo) / levels as f32;
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(lo).to_le_bytes());
        // Re-read the stored parameters so quantisation matches what dequant
        // will see; otherwise f16 rounding shows up as a systematic bias.
        let d_s = f16_to_f32(f32_to_f16(d));
        let lo_s = f16_to_f32(f32_to_f16(lo));
        let inv = if d_s > 0.0 { 1.0 / d_s } else { 0.0 };
        for (qi, &v) in q.iter_mut().zip(g) {
            *qi = (((v - lo_s) * inv).round()).clamp(0.0, levels as f32) as u8;
        }
        pack_bits(&q, ty.payload_bits(), out);
    }
    Ok(())
}

/// Dequantise `n` weights of `ty` from `src` into `out`.
pub fn dequantize(
    ty: PmxType,
    src: &[u8],
    n: usize,
    out: &mut Vec<f32>,
) -> Result<(), KernelError> {
    let need = ty.bytes_for(n)?;
    if src.len() < need {
        return Err(KernelError::ShortBuffer {
            need,
            have: src.len(),
        });
    }
    out.clear();
    out.resize(n, 0.0);
    let gb = ty.group_bytes();
    let bits = ty.payload_bits();
    let mut q = vec![0u8; GROUP];

    for (gi, o) in out.chunks_exact_mut(GROUP).enumerate() {
        let b = &src[gi * gb..(gi + 1) * gb];
        if ty == PmxType::Q8 {
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for (j, ov) in o.iter_mut().enumerate() {
                *ov = (b[2 + j] as i8) as f32 * d;
            }
            continue;
        }
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let lo = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        unpack_bits(&b[4..], bits, GROUP, &mut q);
        for (ov, &qi) in o.iter_mut().zip(q.iter()) {
            *ov = f32::from(qi) * d + lo;
        }
    }
    Ok(())
}

/// Root-mean-square error of a quantise/dequantise round trip.
///
/// Used to report what a precision choice actually costs on real weights,
/// instead of relying on the analytic proxy in `pmx-plan`.
///
/// Non-finite inputs are skipped rather than propagated. A single `Inf` in a
/// checkpoint would otherwise make this `NaN`, and a `NaN` sensitivity silently
/// disables every allocation decision downstream — comparisons against it are
/// all false, so nothing is ever judged affordable and the allocator reports
/// success while doing nothing. Returns `None` if no finite pair remains.
pub fn roundtrip_rmse(ty: PmxType, x: &[f32]) -> Result<Option<f64>, KernelError> {
    let mut packed = Vec::new();
    quantize(ty, x, &mut packed)?;
    let mut back = Vec::new();
    dequantize(ty, &packed, x.len(), &mut back)?;
    let mut acc = 0.0f64;
    let mut n = 0u64;
    for (a, b) in x.iter().zip(&back) {
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let d = f64::from(*a) - f64::from(*b);
        acc += d * d;
        n += 1;
    }
    if n == 0 {
        return Ok(None);
    }
    Ok(Some((acc / n as f64).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / n as f32) * 4.0 - 2.0).collect()
    }

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

    #[test]
    fn bits_per_weight_matches_the_documented_table() {
        assert_eq!(PmxType::Q8.bits_per_weight(), 8.25);
        assert_eq!(PmxType::Q5.bits_per_weight(), 5.5);
        assert_eq!(PmxType::Q4.bits_per_weight(), 4.5);
        assert_eq!(PmxType::Q3.bits_per_weight(), 3.5);
        assert_eq!(PmxType::Q2.bits_per_weight(), 2.5);
    }

    #[test]
    fn group_bytes_are_consistent_with_bytes_for() {
        for ty in PmxType::ALL {
            assert_eq!(ty.bytes_for(GROUP).unwrap(), ty.group_bytes());
            assert_eq!(ty.bytes_for(GROUP * 7).unwrap(), ty.group_bytes() * 7);
            assert!(ty.bytes_for(GROUP - 1).is_err());
        }
    }

    #[test]
    fn tags_round_trip() {
        for ty in PmxType::ALL {
            assert_eq!(PmxType::from_tag(ty.tag()), Some(ty));
        }
        assert_eq!(PmxType::from_tag(7), None);
    }

    #[test]
    fn bit_packing_round_trips_at_every_width() {
        for bits in [2usize, 3, 4, 5, 8] {
            let levels = (1u32 << bits) - 1;
            let vals: Vec<u8> = (0..GROUP)
                .map(|i| (i as u32 % (levels + 1)) as u8)
                .collect();
            let mut packed = Vec::new();
            pack_bits(&vals, bits, &mut packed);
            let mut back = vec![0u8; GROUP];
            unpack_bits(&packed, bits, GROUP, &mut back);
            assert_eq!(back, vals, "bits={bits}");
        }
    }

    #[test]
    fn more_bits_never_means_more_error() {
        // The whole bit-allocation argument depends on this being monotone.
        let x = noisy(GROUP * 16, 7);
        let mut prev = 0.0f64;
        for ty in PmxType::ALL {
            let e = roundtrip_rmse(ty, &x).unwrap().expect("finite");
            assert!(
                e >= prev - 1e-9,
                "{} gave rmse {e} but a more precise format gave {prev}",
                ty.label()
            );
            prev = e;
        }
    }

    #[test]
    fn error_is_bounded_by_the_quantisation_step() {
        // Asymmetric quantisation of a bounded ramp: worst-case error is half a
        // step, so RMSE must be comfortably under a full step.
        let x = ramp(GROUP * 4);
        let span = 4.0f64;
        for ty in [PmxType::Q5, PmxType::Q4, PmxType::Q3, PmxType::Q2] {
            let step = span / ((1u32 << ty.payload_bits()) - 1) as f64;
            let e = roundtrip_rmse(ty, &x).unwrap().expect("finite");
            assert!(e < step, "{}: rmse {e} exceeded step {step}", ty.label());
        }
    }

    #[test]
    fn q8_is_near_exact_on_smooth_data() {
        let x = ramp(GROUP * 4);
        let e = roundtrip_rmse(PmxType::Q8, &x).unwrap().expect("finite");
        assert!(e < 0.02, "q8 rmse {e} is larger than expected");
    }

    #[test]
    fn constant_groups_round_trip_exactly_enough() {
        // A degenerate group (hi == lo) must not divide by zero or drift.
        let x = vec![0.75f32; GROUP * 2];
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            quantize(ty, &x, &mut packed).unwrap();
            let mut back = Vec::new();
            dequantize(ty, &packed, x.len(), &mut back).unwrap();
            for v in &back {
                assert!((*v - 0.75).abs() < 1e-3, "{}: got {v}", ty.label());
            }
        }
    }

    #[test]
    fn all_zeros_stay_zero() {
        let x = vec![0.0f32; GROUP];
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            quantize(ty, &x, &mut packed).unwrap();
            let mut back = Vec::new();
            dequantize(ty, &packed, GROUP, &mut back).unwrap();
            assert!(back.iter().all(|v| *v == 0.0), "{}", ty.label());
        }
    }

    #[test]
    fn packed_size_is_exactly_as_advertised() {
        let x = noisy(GROUP * 5, 3);
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            quantize(ty, &x, &mut packed).unwrap();
            assert_eq!(
                packed.len(),
                ty.bytes_for(x.len()).unwrap(),
                "{}",
                ty.label()
            );
        }
    }

    #[test]
    fn rmse_skips_non_finite_values_instead_of_returning_nan() {
        let mut x = vec![0.5f32; GROUP];
        x[0] = f32::NAN;
        x[1] = f32::INFINITY;
        let e = roundtrip_rmse(PmxType::Q4, &x).unwrap();
        assert!(
            e.is_some(),
            "some finite values remained, so an error should be reported"
        );
        assert!(e.unwrap().is_finite(), "rmse must never be NaN: {e:?}");
    }

    #[test]
    fn rmse_reports_none_when_nothing_is_finite() {
        let x = vec![f32::NAN; GROUP];
        assert_eq!(roundtrip_rmse(PmxType::Q2, &x).unwrap(), None);
    }

    #[test]
    fn non_finite_input_does_not_panic() {
        let mut x = vec![0.5f32; GROUP];
        x[0] = f32::NAN;
        x[1] = f32::INFINITY;
        for ty in PmxType::ALL {
            let mut packed = Vec::new();
            // Must return cleanly rather than panicking or producing garbage sizes.
            quantize(ty, &x, &mut packed).unwrap();
            assert_eq!(packed.len(), ty.bytes_for(GROUP).unwrap());
        }
    }
}
