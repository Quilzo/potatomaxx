// SPDX-License-Identifier: GPL-2.0-or-later
//! Numeric kernels for potatomaxx.
//!
//! Three concerns live here:
//!
//! * [`ggml_dequant`] decodes GGUF block formats, so a checkpoint's weights can
//!   be read at full precision before being requantised.
//! * [`pmxq`] defines the potatomaxx block formats. These exist because
//!   per-expert mixed precision cannot be expressed in GGUF — a tensor carries a
//!   single `ggml_type` and a MoE layer's experts share a tensor.
//! * [`dot`] provides the fused decode-and-multiply inner loop, with SIMD paths
//!   checked against a scalar reference.
//!
//! This is one of the two crates permitted `unsafe`, for SIMD intrinsics. The
//! scalar implementation is authoritative; every vector path is validated
//! against it in tests.

#![warn(missing_docs)]
// SIMD intrinsics require `unsafe`. Each block documents the preconditions it
// relies on, and `dot::detect_isa` is the only thing that decides which path
// runs, so the CPU-feature precondition is established in one place.
#![allow(unsafe_code)]

pub mod dot;
pub mod ggml_dequant;
pub mod half;
pub mod pmxq;

pub use dot::{detect_isa, Isa};
pub use pmxq::{PmxType, GROUP};

use std::fmt;

/// Anything that can go wrong in a kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// The element count is not a whole number of blocks.
    PartialBlock {
        /// Elements requested.
        n_elements: usize,
        /// Elements per block.
        block_elems: usize,
    },
    /// The source buffer is smaller than the blocks it must contain.
    ShortBuffer {
        /// Bytes required.
        need: usize,
        /// Bytes available.
        have: usize,
    },
    /// A GGML type this build cannot size.
    UnknownGgmlType(u32),
    /// A GGML type this build can size but not decode.
    UnsupportedForDequant(u32),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::PartialBlock {
                n_elements,
                block_elems,
            } => write!(
                f,
                "{n_elements} elements is not a multiple of the {block_elems}-element block"
            ),
            KernelError::ShortBuffer { need, have } => {
                write!(f, "buffer holds {have} bytes but {need} are needed")
            }
            KernelError::UnknownGgmlType(t) => write!(f, "unknown GGML type {t}"),
            KernelError::UnsupportedForDequant(t) => write!(
                f,
                "GGML type {t} is not decodable by this build; refusing to guess its layout"
            ),
        }
    }
}

impl std::error::Error for KernelError {}

/// Elements and bytes per block for a GGML type.
///
/// Mirrors `pmx_gguf::GgmlType::block_shape`; duplicated so this crate stays
/// independent of the container parser.
pub fn block_shape(ty: u32) -> Result<(usize, usize), KernelError> {
    let v = match ty {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        other => return Err(KernelError::UnknownGgmlType(other)),
    };
    Ok(v)
}

/// Whether this build can decode `ty`.
pub fn can_dequantize(ty: u32) -> bool {
    matches!(ty, 0 | 1 | 2 | 3 | 8 | 12 | 14 | 30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_shape_agrees_with_the_container_parser_table() {
        assert_eq!(block_shape(12).unwrap(), (256, 144));
        assert_eq!(block_shape(14).unwrap(), (256, 210));
        assert_eq!(block_shape(8).unwrap(), (32, 34));
        assert!(block_shape(9999).is_err());
    }

    #[test]
    fn decodable_types_are_exactly_those_dequantize_handles() {
        for ty in 0u32..40 {
            if block_shape(ty).is_err() {
                continue;
            }
            let (be, bb) = block_shape(ty).unwrap();
            let src = vec![0u8; bb * 2];
            let r = ggml_dequant::dequantize(ty, &src, be * 2, &mut Vec::new());
            assert_eq!(
                r.is_ok(),
                can_dequantize(ty),
                "type {ty}: can_dequantize says {} but dequantize returned {:?}",
                can_dequantize(ty),
                r
            );
        }
    }
}
