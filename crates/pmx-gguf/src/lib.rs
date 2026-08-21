//! Strict, zero-dependency GGUF reading and offset-rewriting.
//!
//! # Why this exists
//!
//! `potatomaxx` rewrites the *physical layout* of a GGUF file: the same tensors,
//! with the same names and byte-identical contents, placed at different offsets
//! so that experts which fire together land in the same 2 MiB read.
//!
//! The GGUF specification permits exactly this. Tensor data is located solely
//! through the `offset` field of its `tensor_info` record; the spec does not
//! require the data section to follow header order, and explicitly allows
//! padding between tensors. So a repacked file is a *drop-in* GGUF.
//!
//! # Why it is written this way
//!
//! Model files are untrusted input downloaded from public hubs, and the parser
//! is the part of an inference stack with no performance requirement at all.
//! The recent history of this format is a run of memory-safety failures in
//! exactly this code path:
//!
//! - **CVE-2026-27940** — integer overflow in `gguf_init_from_file_impl()`
//!   producing an undersized heap allocation, then a 528+ byte controlled
//!   overflow. Itself a bypass of the fix for CVE-2025-53630.
//! - **CVE-2026-7482** ("Bleeding Llama", CVSS 9.1) — out-of-bounds read from
//!   inflated tensor dimensions, leaking process memory.
//!
//! Accordingly this crate forbids `unsafe`, performs every length and offset
//! computation with checked arithmetic, and validates each record against the
//! real file size before it is trusted. Malformed input yields a typed error,
//! never a panic and never a bad read.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ggml;
mod read;
mod write;

pub use ggml::GgmlType;
pub use read::{Gguf, MetaValue, TensorInfo, ValueType};
pub use write::{repack, verify, PermSpec, Placement, RepackReport, VerifyReport};

use std::fmt;

/// The four-byte file magic: `GGUF`.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// Default alignment when a file carries no `general.alignment` key.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Anything that can go wrong reading or rewriting a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The file did not begin with `GGUF`.
    BadMagic([u8; 4]),
    /// Unsupported container version.
    UnsupportedVersion(u32),
    /// A read ran past the end of the header region.
    Truncated {
        /// What the parser was trying to read.
        what: &'static str,
        /// Bytes it needed.
        need: u64,
        /// Bytes actually available.
        have: u64,
    },
    /// A declared count or length was larger than the file could possibly hold.
    ImplausibleCount {
        /// What the count described.
        what: &'static str,
        /// The declared value.
        value: u64,
        /// The ceiling this build enforces.
        limit: u64,
    },
    /// A metadata value carried an unrecognised type tag.
    UnknownValueType(u32),
    /// A tensor carried an unrecognised GGML type id.
    UnknownTensorType(u32),
    /// A tensor's element count is not a whole number of blocks.
    MisalignedTensor {
        /// Declared element count.
        n_elements: u64,
        /// Block size required by the tensor's type.
        block_elems: u64,
    },
    /// A tensor's data range falls outside the file.
    TensorOutOfBounds {
        /// Tensor name.
        name: String,
        /// Byte offset within the data section.
        offset: u64,
        /// Byte length.
        len: u64,
        /// Size of the data section.
        data_len: u64,
    },
    /// A tensor offset violated the file's declared alignment.
    UnalignedOffset {
        /// Tensor name.
        name: String,
        /// The offending offset.
        offset: u64,
        /// Required alignment.
        alignment: u64,
    },
    /// A string field was not valid UTF-8.
    BadUtf8(&'static str),
    /// A checked arithmetic operation overflowed.
    ArithmeticOverflow(&'static str),
    /// A repack plan did not describe every tensor exactly once.
    InvalidPlan(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {e}"),
            GgufError::BadMagic(m) => {
                write!(f, "not a GGUF file (magic was {m:?}, expected \"GGUF\")")
            }
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Truncated { what, need, have } => {
                write!(
                    f,
                    "truncated while reading {what}: needed {need} bytes, {have} available"
                )
            }
            GgufError::ImplausibleCount { what, value, limit } => {
                write!(
                    f,
                    "implausible {what} count {value} (limit {limit}); refusing to allocate"
                )
            }
            GgufError::UnknownValueType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::UnknownTensorType(t) => write!(f, "unknown GGML tensor type {t}"),
            GgufError::MisalignedTensor {
                n_elements,
                block_elems,
            } => write!(
                f,
                "tensor has {n_elements} elements, not a multiple of block size {block_elems}"
            ),
            GgufError::TensorOutOfBounds {
                name,
                offset,
                len,
                data_len,
            } => write!(
                f,
                "tensor {name:?} spans {offset}..{} but the data section is only {data_len} bytes",
                offset.saturating_add(*len)
            ),
            GgufError::UnalignedOffset {
                name,
                offset,
                alignment,
            } => write!(
                f,
                "tensor {name:?} offset {offset} is not a multiple of alignment {alignment}"
            ),
            GgufError::BadUtf8(what) => write!(f, "{what} was not valid UTF-8"),
            GgufError::ArithmeticOverflow(what) => {
                write!(f, "arithmetic overflow computing {what}")
            }
            GgufError::InvalidPlan(m) => write!(f, "invalid repack plan: {m}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// Round `n` up to the next multiple of `align`.
///
/// `align` of 0 is treated as 1. Returns `None` on overflow rather than
/// wrapping — a wrapped alignment is how you get an undersized allocation.
pub fn align_up(n: u64, align: u64) -> Option<u64> {
    if align <= 1 {
        return Some(n);
    }
    let rem = n % align;
    if rem == 0 {
        Some(n)
    } else {
        n.checked_add(align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basics() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(5, 1), Some(5));
        assert_eq!(align_up(5, 0), Some(5));
    }

    #[test]
    fn align_up_refuses_to_wrap() {
        assert_eq!(align_up(u64::MAX, 4096), None);
    }

    #[test]
    fn tensor_bytes_matches_known_block_layouts() {
        // Q4_K: 256 elements per 144-byte block.
        assert_eq!(GgmlType(12).tensor_bytes(256).unwrap(), 144);
        assert_eq!(GgmlType(12).tensor_bytes(2560).unwrap(), 1440);
        // F32: 4 bytes per element.
        assert_eq!(GgmlType(0).tensor_bytes(10).unwrap(), 40);
        // Q6_K: 256 elements per 210-byte block.
        assert_eq!(GgmlType(14).tensor_bytes(512).unwrap(), 420);
    }

    #[test]
    fn tensor_bytes_rejects_partial_blocks() {
        assert!(matches!(
            GgmlType(12).tensor_bytes(255),
            Err(GgufError::MisalignedTensor { .. })
        ));
    }

    #[test]
    fn unknown_tensor_type_is_an_error_not_a_guess() {
        assert!(matches!(
            GgmlType(9999).tensor_bytes(256),
            Err(GgufError::UnknownTensorType(9999))
        ));
    }
}
