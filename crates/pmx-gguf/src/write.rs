// SPDX-License-Identifier: GPL-2.0-or-later
//! Rewriting a GGUF file's physical layout.
//!
//! # The two mechanisms
//!
//! **Tensor reordering.** Tensor data is located only through the `offset` in
//! its header record, so the physical order of the data section is free. We
//! keep the `tensor_info` records in their original order (so a loader
//! iterating them sees an unchanged sequence) and change only their offsets.
//!
//! **Expert-slot permutation.** In practice a GGUF MoE checkpoint does not
//! store one tensor per expert; it stacks every expert of a layer into a single
//! tensor whose last dimension is the expert axis, e.g.
//! `blk.7.ffn_up_exps.weight` with dims `[n_embd, n_ff, n_expert]`. Locality
//! therefore has to be created *inside* that tensor, by permuting the slices
//! along the expert axis.
//!
//! Permuting experts is sound because it is only a relabelling. If new slot `j`
//! holds old expert `perm[j]`, and the router's weight rows are permuted by the
//! same `perm`, then the logit computed for slot `j` is exactly the logit the
//! original model computed for expert `perm[j]`. Top-k therefore selects the
//! same set of real experts, and each selected slot contains that expert's
//! original bytes. The function computed is unchanged, bit for bit.
//!
//! It is the caller's responsibility to supply a permutation for *every* tensor
//! carrying the expert axis of a given layer — the expert tensors and the
//! router. [`verify`] exists to prove that afterwards.

use crate::{align_up, read::TensorInfo, Gguf, GgufError, MAGIC};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Bytes copied per read/write step when streaming a tensor unchanged.
const COPY_CHUNK: usize = 4 << 20;

/// A permutation applied along a tensor's last (expert) axis.
#[derive(Debug, Clone)]
pub struct PermSpec {
    /// Name of the tensor to permute.
    pub tensor: String,
    /// `perm[j] == e` means new slot `j` receives old slice `e`.
    pub perm: Vec<u32>,
}

/// A complete description of the layout to emit.
#[derive(Debug, Clone, Default)]
pub struct Placement {
    /// Physical write order as indices into [`Gguf::tensors`]. Must be a
    /// permutation of `0..n`. Empty means "keep the original order".
    pub order: Vec<usize>,
    /// Alignment applied where a new group begins, in bytes. 0 or 1 disables.
    pub group_align: u64,
    /// Positions *within `order`* at which a new group begins.
    pub group_starts: HashSet<usize>,
    /// Expert-axis permutations, keyed by tensor name.
    pub perms: Vec<PermSpec>,
}

/// What a repack actually did.
#[derive(Debug, Clone)]
pub struct RepackReport {
    /// Bytes of tensor data written.
    pub data_bytes: u64,
    /// Bytes of alignment padding inserted inside the data section.
    pub padding_bytes: u64,
    /// Number of groups the data section was divided into.
    pub groups: usize,
    /// Header size of the emitted file.
    pub header_bytes: u64,
    /// Absolute offset of the emitted data section.
    pub data_start: u64,
    /// Tensors whose expert axis was permuted.
    pub permuted_tensors: usize,
}

/// Slice geometry of a tensor along its last axis.
struct SliceGeom {
    /// Number of slices (the last dimension).
    n_slices: u64,
    /// Bytes per slice.
    slice_bytes: u64,
}

fn slice_geometry(t: &TensorInfo) -> Result<SliceGeom, GgufError> {
    let n_slices = *t.dims.last().unwrap_or(&1);
    if n_slices == 0 {
        return Err(GgufError::InvalidPlan(format!(
            "tensor {:?} has a zero-length last dimension",
            t.name
        )));
    }
    if t.nbytes % n_slices != 0 {
        return Err(GgufError::InvalidPlan(format!(
            "tensor {:?} ({} bytes) does not divide evenly into {} expert slices; \
             its quantisation blocks straddle the expert axis and it cannot be permuted safely",
            t.name, t.nbytes, n_slices
        )));
    }
    Ok(SliceGeom {
        n_slices,
        slice_bytes: t.nbytes / n_slices,
    })
}

fn validate(src: &Gguf, plan: &Placement) -> Result<Vec<usize>, GgufError> {
    let n = src.tensors.len();
    let order: Vec<usize> = if plan.order.is_empty() {
        (0..n).collect()
    } else {
        if plan.order.len() != n {
            return Err(GgufError::InvalidPlan(format!(
                "order lists {} tensors but the file has {}",
                plan.order.len(),
                n
            )));
        }
        let mut seen = vec![false; n];
        for &i in &plan.order {
            if i >= n {
                return Err(GgufError::InvalidPlan(format!(
                    "order references tensor {i}, out of range"
                )));
            }
            if seen[i] {
                return Err(GgufError::InvalidPlan(format!(
                    "order lists tensor {i} twice"
                )));
            }
            seen[i] = true;
        }
        plan.order.clone()
    };

    // Every permutation must name a real tensor and be a true permutation of
    // that tensor's expert axis.
    let by_name: HashMap<&str, &TensorInfo> =
        src.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    for p in &plan.perms {
        let t = by_name.get(p.tensor.as_str()).ok_or_else(|| {
            GgufError::InvalidPlan(format!("permutation names unknown tensor {:?}", p.tensor))
        })?;
        let g = slice_geometry(t)?;
        if p.perm.len() as u64 != g.n_slices {
            return Err(GgufError::InvalidPlan(format!(
                "permutation for {:?} has {} entries but the expert axis is {}",
                p.tensor,
                p.perm.len(),
                g.n_slices
            )));
        }
        let mut seen = vec![false; p.perm.len()];
        for &e in &p.perm {
            let e = e as usize;
            if e >= seen.len() {
                return Err(GgufError::InvalidPlan(format!(
                    "permutation for {:?} references slice {e}, out of range",
                    p.tensor
                )));
            }
            if seen[e] {
                return Err(GgufError::InvalidPlan(format!(
                    "permutation for {:?} references slice {e} twice; it is not a permutation",
                    p.tensor
                )));
            }
            seen[e] = true;
        }
    }
    Ok(order)
}

/// Serialise the header, given final offsets, and return the bytes.
fn build_header(src: &Gguf, offsets: &[u64]) -> Result<Vec<u8>, GgufError> {
    let mut h = Vec::with_capacity(1 << 16);
    h.extend_from_slice(&MAGIC);
    h.extend_from_slice(&src.version.to_le_bytes());
    h.extend_from_slice(&(src.tensors.len() as u64).to_le_bytes());
    h.extend_from_slice(&(src.kvs.len() as u64).to_le_bytes());

    for (k, v) in &src.kvs {
        h.extend_from_slice(&(k.len() as u64).to_le_bytes());
        h.extend_from_slice(k.as_bytes());
        h.extend_from_slice(&v.tag().to_le_bytes());
        h.extend_from_slice(&v.raw);
    }
    // tensor_info records stay in their original order; only offsets change.
    for (t, off) in src.tensors.iter().zip(offsets) {
        h.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        h.extend_from_slice(t.name.as_bytes());
        h.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            h.extend_from_slice(&d.to_le_bytes());
        }
        h.extend_from_slice(&t.ggml_type.0.to_le_bytes());
        h.extend_from_slice(&off.to_le_bytes());
    }
    Ok(h)
}

/// Rewrite `src` into `out` according to `plan`.
///
/// Tensor *contents* are preserved exactly; only their position in the file and
/// the order of slices along permuted expert axes change.
pub fn repack(
    src: &Gguf,
    plan: &Placement,
    out: impl AsRef<Path>,
) -> Result<RepackReport, GgufError> {
    let order = validate(src, plan)?;
    let perms: HashMap<&str, &Vec<u32>> = plan
        .perms
        .iter()
        .map(|p| (p.tensor.as_str(), &p.perm))
        .collect();
    let group_align = if plan.group_align <= 1 {
        0
    } else {
        plan.group_align
    };

    // Pass 1: lay out offsets relative to the data section.
    let mut offsets = vec![0u64; src.tensors.len()];
    let mut cursor: u64 = 0;
    let mut padding: u64 = 0;
    let mut groups = if order.is_empty() { 0 } else { 1 };
    for (pos, &ti) in order.iter().enumerate() {
        let want_group = group_align > 0 && pos > 0 && plan.group_starts.contains(&pos);
        if want_group {
            let bumped = align_up(cursor, group_align)
                .ok_or(GgufError::ArithmeticOverflow("group alignment"))?;
            padding += bumped - cursor;
            cursor = bumped;
            groups += 1;
        } else {
            let bumped = align_up(cursor, src.alignment)
                .ok_or(GgufError::ArithmeticOverflow("tensor alignment"))?;
            padding += bumped - cursor;
            cursor = bumped;
        }
        offsets[ti] = cursor;
        cursor = cursor
            .checked_add(src.tensors[ti].nbytes)
            .ok_or(GgufError::ArithmeticOverflow("data section length"))?;
    }
    let data_bytes = cursor;

    // Pass 2: emit. The header must be padded so the data section starts on an
    // alignment boundary; when grouping, also align the section itself so that
    // group offsets are absolute multiples of `group_align`.
    let header = build_header(src, &offsets)?;
    let header_len = header.len() as u64;
    let section_align = if group_align > 0 {
        group_align.max(src.alignment)
    } else {
        src.alignment
    };
    let data_start =
        align_up(header_len, section_align).ok_or(GgufError::ArithmeticOverflow("data start"))?;

    let mut w = BufWriter::with_capacity(1 << 20, File::create(out.as_ref())?);
    w.write_all(&header)?;
    let zeros = vec![
        0u8;
        usize::try_from(section_align.max(1))
            .unwrap_or(1)
            .min(1 << 20)
    ];
    write_zeros(&mut w, data_start - header_len, &zeros)?;

    let mut src_f = File::open(&src.path)?;
    let mut written: u64 = 0;
    let mut permuted = 0usize;
    for &ti in &order {
        let t = &src.tensors[ti];
        let target = offsets[ti];
        debug_assert!(target >= written);
        write_zeros(&mut w, target - written, &zeros)?;
        written = target;

        let abs = src
            .data_start
            .checked_add(t.offset)
            .ok_or(GgufError::ArithmeticOverflow("source tensor offset"))?;

        match perms.get(t.name.as_str()) {
            Some(perm) => {
                let g = slice_geometry(t)?;
                let mut buf = vec![
                    0u8;
                    usize::try_from(g.slice_bytes).map_err(|_| {
                        GgufError::ArithmeticOverflow("expert slice length")
                    })?
                ];
                for &old in perm.iter() {
                    let so = abs
                        .checked_add(u64::from(old) * g.slice_bytes)
                        .ok_or(GgufError::ArithmeticOverflow("slice offset"))?;
                    src_f.seek(SeekFrom::Start(so))?;
                    src_f.read_exact(&mut buf)?;
                    w.write_all(&buf)?;
                }
                permuted += 1;
            }
            None => {
                src_f.seek(SeekFrom::Start(abs))?;
                copy_exact(&mut src_f, &mut w, t.nbytes)?;
            }
        }
        written += t.nbytes;
    }
    w.flush()?;

    Ok(RepackReport {
        data_bytes,
        padding_bytes: padding,
        groups,
        header_bytes: header_len,
        data_start,
        permuted_tensors: permuted,
    })
}

fn write_zeros<W: Write>(w: &mut W, mut n: u64, zeros: &[u8]) -> Result<(), GgufError> {
    while n > 0 {
        let step = usize::try_from(n).unwrap_or(zeros.len()).min(zeros.len());
        w.write_all(&zeros[..step])?;
        n -= step as u64;
    }
    Ok(())
}

fn copy_exact<R: Read, W: Write>(r: &mut R, w: &mut W, mut n: u64) -> Result<(), GgufError> {
    let mut buf = vec![
        0u8;
        COPY_CHUNK
            .min(usize::try_from(n).unwrap_or(COPY_CHUNK))
            .max(1)
    ];
    while n > 0 {
        let step = usize::try_from(n).unwrap_or(buf.len()).min(buf.len());
        r.read_exact(&mut buf[..step])?;
        w.write_all(&buf[..step])?;
        n -= step as u64;
    }
    Ok(())
}

/// The outcome of comparing a repacked file against its source.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Tensors compared.
    pub tensors: usize,
    /// Tensors whose bytes were found identical without any permutation.
    pub identical: usize,
    /// Tensors that matched as a permutation of the original slices.
    pub permuted: usize,
    /// Total bytes compared.
    pub bytes: u64,
}

/// Prove that `dst` contains exactly the weights of `src`.
///
/// Every tensor is matched by name and must agree on type and dimensions. Its
/// content must be either byte-identical, or — where `perms` supplies a
/// permutation — equal to the source slices reordered by it. Any other
/// difference is an error.
///
/// This is the check that backs the claim "same weights, same outputs". It
/// reads both files in full.
pub fn verify(src: &Gguf, dst: &Gguf, perms: &[PermSpec]) -> Result<VerifyReport, GgufError> {
    let perms: HashMap<&str, &Vec<u32>> =
        perms.iter().map(|p| (p.tensor.as_str(), &p.perm)).collect();
    let dst_by_name: HashMap<&str, &TensorInfo> =
        dst.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut rep = VerifyReport {
        tensors: 0,
        identical: 0,
        permuted: 0,
        bytes: 0,
    };
    let mut sf = File::open(&src.path)?;
    let mut df = File::open(&dst.path)?;

    for st in &src.tensors {
        let dt = dst_by_name.get(st.name.as_str()).ok_or_else(|| {
            GgufError::InvalidPlan(format!("repacked file is missing tensor {:?}", st.name))
        })?;
        if dt.ggml_type != st.ggml_type || dt.dims != st.dims || dt.nbytes != st.nbytes {
            return Err(GgufError::InvalidPlan(format!(
                "tensor {:?} changed shape or type: {:?}/{} -> {:?}/{}",
                st.name,
                st.dims,
                st.ggml_type.name(),
                dt.dims,
                dt.ggml_type.name()
            )));
        }
        let g = slice_geometry(st)?;
        let perm: Vec<u32> = match perms.get(st.name.as_str()) {
            Some(p) => (*p).clone(),
            None => (0..g.n_slices as u32).collect(),
        };
        let is_identity = perm.iter().enumerate().all(|(j, &e)| j as u32 == e);

        let mut a = vec![
            0u8;
            usize::try_from(g.slice_bytes)
                .map_err(|_| GgufError::ArithmeticOverflow("slice length"))?
        ];
        let mut b = a.clone();
        for (new_slot, &old) in perm.iter().enumerate() {
            sf.seek(SeekFrom::Start(
                src.data_start + st.offset + u64::from(old) * g.slice_bytes,
            ))?;
            sf.read_exact(&mut a)?;
            df.seek(SeekFrom::Start(
                dst.data_start + dt.offset + new_slot as u64 * g.slice_bytes,
            ))?;
            df.read_exact(&mut b)?;
            if a != b {
                return Err(GgufError::InvalidPlan(format!(
                    "tensor {:?} slice {new_slot} does not match source slice {old}",
                    st.name
                )));
            }
        }
        rep.tensors += 1;
        rep.bytes += st.nbytes;
        if is_identity {
            rep.identical += 1;
        } else {
            rep.permuted += 1;
        }
    }
    Ok(rep)
}
