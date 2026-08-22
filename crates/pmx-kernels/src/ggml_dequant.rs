// SPDX-License-Identifier: GPL-2.0-or-later
//! Dequantising GGUF block formats to `f32`.
//!
//! These must match ggml's `dequantize_row_*` bit for bit: everything
//! downstream — requantisation, sensitivity estimates, predicted quality — is
//! wrong if a super-block is decoded wrong. The layouts are transcribed from
//! ggml's block structs, and each has a test against values produced by
//! constructing a block whose expected output is known analytically.
//!
//! Formats not implemented here return [`KernelError::UnsupportedForDequant`]
//! rather than a guess. A silently wrong dequantiser would be far worse than a
//! refusal, because the resulting model would load and produce plausible
//! garbage.

use crate::half::f16_to_f32;
use crate::KernelError;

/// Super-block size shared by every k-quant format.
pub const QK_K: usize = 256;

fn rd_f16(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Dequantise `n_elements` of GGML type `ty` from `src` into `out`.
///
/// `out` is resized to `n_elements`.
pub fn dequantize(
    ty: u32,
    src: &[u8],
    n_elements: usize,
    out: &mut Vec<f32>,
) -> Result<(), KernelError> {
    let (block_elems, block_bytes) = crate::block_shape(ty)?;
    if n_elements % block_elems != 0 {
        return Err(KernelError::PartialBlock {
            n_elements,
            block_elems,
        });
    }
    let nblocks = n_elements / block_elems;
    let need = nblocks * block_bytes;
    if src.len() < need {
        return Err(KernelError::ShortBuffer {
            need,
            have: src.len(),
        });
    }
    out.clear();
    out.resize(n_elements, 0.0);

    match ty {
        0 => {
            // F32
            for (i, o) in out.iter_mut().enumerate() {
                let b = &src[i * 4..i * 4 + 4];
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        1 => {
            // F16
            for (i, o) in out.iter_mut().enumerate() {
                *o = rd_f16(src, i * 2);
            }
        }
        30 => {
            // BF16: the top 16 bits of an f32.
            for (i, o) in out.iter_mut().enumerate() {
                let hi = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                *o = f32::from_bits(u32::from(hi) << 16);
            }
        }
        2 => dequant_q4_0(src, nblocks, out),
        3 => dequant_q4_1(src, nblocks, out),
        8 => dequant_q8_0(src, nblocks, out),
        12 => dequant_q4_k(src, nblocks, out),
        14 => dequant_q6_k(src, nblocks, out),
        other => return Err(KernelError::UnsupportedForDequant(other)),
    }
    Ok(())
}

/// Q4_0: `f16 d`, then 16 bytes holding 32 four-bit quants.
///
/// Low nibble of byte `j` is element `j`; high nibble is element `j + 16`.
fn dequant_q4_0(src: &[u8], nblocks: usize, out: &mut [f32]) {
    for i in 0..nblocks {
        let b = &src[i * 18..(i + 1) * 18];
        let d = rd_f16(b, 0);
        let qs = &b[2..18];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            out[i * 32 + j] = x0 as f32 * d;
            out[i * 32 + j + 16] = x1 as f32 * d;
        }
    }
}

/// Q4_1: `f16 d`, `f16 m`, then 16 bytes of 32 four-bit quants. `y = d*q + m`.
fn dequant_q4_1(src: &[u8], nblocks: usize, out: &mut [f32]) {
    for i in 0..nblocks {
        let b = &src[i * 20..(i + 1) * 20];
        let d = rd_f16(b, 0);
        let m = rd_f16(b, 2);
        let qs = &b[4..20];
        for j in 0..16 {
            out[i * 32 + j] = (qs[j] & 0x0F) as f32 * d + m;
            out[i * 32 + j + 16] = (qs[j] >> 4) as f32 * d + m;
        }
    }
}

/// Q8_0: `f16 d`, then 32 signed bytes. `y = d*q`.
fn dequant_q8_0(src: &[u8], nblocks: usize, out: &mut [f32]) {
    for i in 0..nblocks {
        let b = &src[i * 34..(i + 1) * 34];
        let d = rd_f16(b, 0);
        for j in 0..32 {
            out[i * 32 + j] = (b[2 + j] as i8) as f32 * d;
        }
    }
}

/// Unpack one of the eight 6-bit scale/min pairs from a Q4_K super-block.
///
/// This is ggml's `get_scale_min_k4`. The packing is irregular: the first four
/// pairs are stored plainly in the low six bits, and the last four borrow their
/// high two bits from the top of the first four bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K: `f16 d`, `f16 dmin`, 12 bytes of packed 6-bit scales/mins, 128 bytes of
/// 4-bit quants. Eight sub-blocks of 32, each with its own scale and min.
fn dequant_q4_k(src: &[u8], nblocks: usize, out: &mut [f32]) {
    for i in 0..nblocks {
        let b = &src[i * 144..(i + 1) * 144];
        let d = rd_f16(b, 0);
        let dmin = rd_f16(b, 2);
        let scales = &b[4..16];
        let qs = &b[16..144];

        let mut y = i * QK_K;
        let mut is = 0usize;
        let mut q = 0usize;
        while is < 8 {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            for l in 0..32 {
                out[y + l] = d1 * (qs[q + l] & 0x0F) as f32 - mm1;
            }
            for l in 0..32 {
                out[y + 32 + l] = d2 * (qs[q + l] >> 4) as f32 - mm2;
            }
            y += 64;
            q += 32;
            is += 2;
        }
    }
}

/// Q6_K: 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 signed
/// 8-bit scales, then `f16 d`. Quants are biased by -32.
// The `>> 0` below is deliberate: it keeps the 0/2/4/6 shift progression visible
// against ggml's reference implementation, which is what this must match.
#[allow(clippy::identity_op)]
fn dequant_q6_k(src: &[u8], nblocks: usize, out: &mut [f32]) {
    for i in 0..nblocks {
        let b = &src[i * 210..(i + 1) * 210];
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = rd_f16(b, 208);

        let base = i * QK_K;
        // Two halves of 128 elements each.
        for n in 0..2 {
            let ql = &ql[n * 64..];
            let qh = &qh[n * 32..];
            let sc = &sc[n * 8..];
            let y = base + n * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                out[y + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                out[y + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                out[y + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                out[y + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::f32_to_f16;

    #[test]
    fn f32_passthrough() {
        let vals = [1.0f32, -2.5, 0.0, 1e10];
        let mut src = Vec::new();
        for v in vals {
            src.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = Vec::new();
        dequantize(0, &src, 4, &mut out).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn f16_and_bf16() {
        let mut src = Vec::new();
        src.extend_from_slice(&f32_to_f16(1.5).to_le_bytes());
        src.extend_from_slice(&f32_to_f16(-0.25).to_le_bytes());
        let mut out = Vec::new();
        dequantize(1, &src, 2, &mut out).unwrap();
        assert_eq!(out, vec![1.5, -0.25]);

        // BF16 keeps the top 16 bits of the f32.
        let mut src = Vec::new();
        for v in [1.5f32, -0.25] {
            src.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut out = Vec::new();
        dequantize(30, &src, 2, &mut out).unwrap();
        assert_eq!(out, vec![1.5, -0.25]);
    }

    #[test]
    fn q8_0_scales_signed_bytes() {
        let mut src = vec![0u8; 34];
        src[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        for j in 0..32 {
            src[2 + j] = (j as i8 - 16) as u8;
        }
        let mut out = Vec::new();
        dequantize(8, &src, 32, &mut out).unwrap();
        for (j, v) in out.iter().enumerate() {
            assert_eq!(*v, (j as i32 - 16) as f32 * 0.5, "element {j}");
        }
    }

    #[test]
    fn q4_0_biases_by_eight_and_splits_nibbles() {
        let mut src = vec![0u8; 18];
        src[0..2].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        // Byte 0: low nibble 0 (-> element 0), high nibble 15 (-> element 16).
        src[2] = 0xF0;
        let mut out = Vec::new();
        dequantize(2, &src, 32, &mut out).unwrap();
        assert_eq!(out[0], (0 - 8) as f32 * 2.0);
        assert_eq!(out[16], (15 - 8) as f32 * 2.0);
    }

    #[test]
    fn q4_1_applies_an_offset() {
        let mut src = vec![0u8; 20];
        src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        src[2..4].copy_from_slice(&f32_to_f16(-3.0).to_le_bytes());
        src[4] = 0x51; // low 1, high 5
        let mut out = Vec::new();
        dequantize(3, &src, 32, &mut out).unwrap();
        assert_eq!(out[0], 1.0 * 1.0 - 3.0);
        assert_eq!(out[16], 5.0 * 1.0 - 3.0);
    }

    #[test]
    fn q4_k_scale_min_unpacking_matches_ggml() {
        // Exercise the irregular packing directly with a known bit pattern.
        let mut q = [0u8; 12];
        q[0] = 0b11_000101; // low6 = 5, top2 = 3
        q[4] = 0b11_001010; // low6 = 10, top2 = 3
        q[8] = 0b0110_1001;
        assert_eq!(scale_min_k4(0, &q), (5, 10));
        // j = 4 borrows high bits from q[0] and q[4].
        let (d, m) = scale_min_k4(4, &q);
        assert_eq!(d, (q[8] & 0x0F) | ((q[0] >> 6) << 4));
        assert_eq!(m, (q[8] >> 4) | ((q[4] >> 6) << 4));
    }

    #[test]
    fn q4_k_decodes_a_uniform_super_block() {
        // d = 1, dmin = 0, every scale = 1: output equals the raw nibbles.
        let mut src = vec![0u8; 144];
        src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        src[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes());
        for b in src[4..8].iter_mut() {
            *b = 1; // scales 0..3 = 1
        }
        for b in src[8..12].iter_mut() {
            *b = 0; // mins 0..3 = 0
        }
        // Pairs 4..7 read from bytes 8..11 of the scale block; set them to 1.
        for b in src[12..16].iter_mut() {
            *b = 0x01;
        }
        for b in src[16..144].iter_mut() {
            *b = 0x21; // low nibble 1, high nibble 2
        }
        let mut out = Vec::new();
        dequantize(12, &src, 256, &mut out).unwrap();
        // Every element is either 1*1 or 2*1 depending on which nibble it came from.
        assert!(
            out.iter().all(|v| *v == 1.0 || *v == 2.0),
            "got {:?}",
            &out[..8]
        );
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn q6_k_decodes_with_the_minus_32_bias() {
        let mut src = vec![0u8; 210];
        // ql all zero, qh all zero -> raw quant 0 -> value -32.
        for b in src[192..208].iter_mut() {
            *b = 1; // all scales = 1
        }
        src[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let mut out = Vec::new();
        dequantize(14, &src, 256, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == -32.0), "got {:?}", &out[..8]);
    }

    #[test]
    fn unsupported_type_is_refused_not_guessed() {
        let src = vec![0u8; 4096];
        // Q5_K (13) is a real type we deliberately do not decode.
        assert!(matches!(
            dequantize(13, &src, 256, &mut Vec::new()),
            Err(KernelError::UnsupportedForDequant(13))
        ));
    }

    #[test]
    fn short_buffers_and_partial_blocks_are_errors() {
        assert!(matches!(
            dequantize(8, &[0u8; 10], 32, &mut Vec::new()),
            Err(KernelError::ShortBuffer { .. })
        ));
        assert!(matches!(
            dequantize(8, &[0u8; 34], 31, &mut Vec::new()),
            Err(KernelError::PartialBlock { .. })
        ));
    }
}
