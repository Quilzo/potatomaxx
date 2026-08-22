// SPDX-License-Identifier: GPL-2.0-or-later
//! GGML tensor type table.
//!
//! Every entry is `(block_elements, block_bytes)`. A tensor holding `n` elements
//! of a type with block size `b` occupies `n / b * block_bytes` bytes, and `n`
//! must be an exact multiple of `b`.
//!
//! Unknown type ids are rejected rather than guessed: mis-sizing a tensor is
//! exactly the bug class that produced CVE-2026-27940, and a wrong guess here
//! would silently corrupt every offset downstream.

use crate::GgufError;

/// A GGML tensor element type, as stored in a GGUF tensor info record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgmlType(pub u32);

impl GgmlType {
    /// Elements per block, and bytes per block.
    ///
    /// Returns `Err` for type ids this build does not know how to size.
    pub fn block_shape(self) -> Result<(u64, u64), GgufError> {
        let v = match self.0 {
            0 => (1, 4),      // F32
            1 => (1, 2),      // F16
            2 => (32, 18),    // Q4_0
            3 => (32, 20),    // Q4_1
            6 => (32, 22),    // Q5_0
            7 => (32, 24),    // Q5_1
            8 => (32, 34),    // Q8_0
            9 => (32, 36),    // Q8_1
            10 => (256, 84),  // Q2_K
            11 => (256, 110), // Q3_K
            12 => (256, 144), // Q4_K
            13 => (256, 176), // Q5_K
            14 => (256, 210), // Q6_K
            15 => (256, 292), // Q8_K
            16 => (256, 66),  // IQ2_XXS
            17 => (256, 74),  // IQ2_XS
            18 => (256, 98),  // IQ3_XXS
            19 => (256, 50),  // IQ1_S
            20 => (32, 18),   // IQ4_NL
            21 => (256, 110), // IQ3_S
            22 => (256, 82),  // IQ2_S
            23 => (256, 136), // IQ4_XS
            24 => (1, 1),     // I8
            25 => (1, 2),     // I16
            26 => (1, 4),     // I32
            27 => (1, 8),     // I64
            28 => (1, 8),     // F64
            29 => (256, 56),  // IQ1_M
            30 => (1, 2),     // BF16
            34 => (256, 54),  // TQ1_0
            35 => (256, 66),  // TQ2_0
            39 => (32, 17),   // MXFP4
            other => return Err(GgufError::UnknownTensorType(other)),
        };
        Ok(v)
    }

    /// Bits per weight, averaged over a block. Used for reporting only.
    pub fn bits_per_weight(self) -> Result<f64, GgufError> {
        let (elems, bytes) = self.block_shape()?;
        Ok((bytes as f64 * 8.0) / elems as f64)
    }

    /// Human-readable name, or `"type-<id>"` for ids without a label.
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            _ => "type-?",
        }
    }

    /// Number of bytes a tensor of this type with `n_elements` elements occupies.
    pub fn tensor_bytes(self, n_elements: u64) -> Result<u64, GgufError> {
        let (block_elems, block_bytes) = self.block_shape()?;
        if block_elems == 0 || n_elements % block_elems != 0 {
            return Err(GgufError::MisalignedTensor {
                n_elements,
                block_elems,
            });
        }
        (n_elements / block_elems)
            .checked_mul(block_bytes)
            .ok_or(GgufError::ArithmeticOverflow("tensor_bytes"))
    }
}
