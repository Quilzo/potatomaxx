//! The potatomaxx native expert store.
//!
//! # Why a second format exists
//!
//! Everything the layout compiler does to a GGUF keeps it a drop-in GGUF, and
//! that is worth preserving. But **per-expert precision cannot be expressed in
//! GGUF at all**: a tensor carries one `ggml_type`, and a MoE layer's experts
//! share a tensor. Giving expert 7 three bits and expert 8 five bits therefore
//! requires a container we define.
//!
//! That is the whole justification for this format, and it is also why the
//! runtime exists. The two paths are deliberately separate:
//!
//! | path | output | consumed by | gives you |
//! |------|--------|-------------|-----------|
//! | layout only | drop-in GGUF | llama.cpp, Colibri, any engine | fewer, larger reads |
//! | + per-expert precision | `.pmxstore` | `pmx-runtime` | fewer *bytes* |
//!
//! # Layout
//!
//! ```text
//! magic "PMXSTORE"   8 bytes
//! version            u32
//! record_count       u64
//! group_align        u64      alignment applied between co-activation groups
//! index_offset       u64
//! data_offset        u64
//! ... index: one 40-byte record per expert tensor slice ...
//! ... data:  slices, ordered by group, each group aligned ...
//! ```
//!
//! Records are sorted by `(layer, slot, kind)`, so a reader can binary-search
//! and, more importantly, so the slices an expert needs are adjacent on disk.
//!
//! # What is *not* requantised
//!
//! Router weights. Quantisation error in a router perturbs the expert selection
//! itself — the "expert shift" problem — which would invalidate the very trace
//! the precision plan was derived from, and silently change which experts the
//! model uses. Routers stay at source precision, always. [`StoreWriter`] has no
//! API to do otherwise.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pmx_kernels::{PmxType, GROUP};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// File magic.
pub const MAGIC: [u8; 8] = *b"PMXSTORE";
/// Current format version.
pub const VERSION: u32 = 1;
/// Bytes per index record on disk.
///
/// Field order puts the three `u64`s on 8-byte boundaries and leaves six
/// reserved bytes, so the record is a round 40 and can grow without a version
/// bump. The written size and this constant must agree exactly — they drifted
/// once during development, and the symptom was every record after the first
/// decoding as garbage.
const RECORD_BYTES: usize = 40;
/// Header size, before the index.
const HEADER_BYTES: u64 = 8 + 4 + 8 + 8 + 8 + 8;

/// Which matrix of an expert a slice belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Gate projection.
    Gate,
    /// Up projection.
    Up,
    /// Down projection.
    Down,
}

impl Kind {
    /// All kinds, in the order they are stored.
    pub const ALL: [Kind; 3] = [Kind::Gate, Kind::Up, Kind::Down];

    /// Wire tag.
    pub fn tag(self) -> u8 {
        match self {
            Kind::Gate => 0,
            Kind::Up => 1,
            Kind::Down => 2,
        }
    }

    /// Recover from a wire tag.
    pub fn from_tag(t: u8) -> Option<Kind> {
        Some(match t {
            0 => Kind::Gate,
            1 => Kind::Up,
            2 => Kind::Down,
            _ => return None,
        })
    }

    /// Infer from a GGUF tensor name.
    pub fn from_tensor_name(name: &str) -> Option<Kind> {
        if name.contains("ffn_gate_exps") {
            Some(Kind::Gate)
        } else if name.contains("ffn_up_exps") {
            Some(Kind::Up)
        } else if name.contains("ffn_down_exps") {
            Some(Kind::Down)
        } else {
            None
        }
    }

    /// Short label.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Gate => "gate",
            Kind::Up => "up",
            Kind::Down => "down",
        }
    }
}

/// One stored expert slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    /// MoE block index.
    pub layer: u32,
    /// Physical slot in the store. After a layout pass this is the permuted
    /// position, not the original expert id.
    pub slot: u32,
    /// Which matrix.
    pub kind: Kind,
    /// Precision this slice is stored at.
    pub ty: PmxType,
    /// Weights in the slice.
    pub n_weights: u64,
    /// Byte offset from the start of the data section.
    pub offset: u64,
    /// Byte length.
    pub nbytes: u64,
}

/// Anything that can go wrong with a store.
#[derive(Debug)]
pub enum StoreError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// Wrong magic.
    BadMagic,
    /// Unsupported version.
    UnsupportedVersion(u32),
    /// A header field described something impossible.
    BadHeader(String),
    /// An index record was malformed.
    BadRecord(String),
    /// A record's byte range falls outside the file.
    OutOfBounds {
        /// Offset within the data section.
        offset: u64,
        /// Length.
        len: u64,
        /// Size of the data section.
        data_len: u64,
    },
    /// A kernel-level failure.
    Kernel(pmx_kernels::KernelError),
    /// A slice's weight count is not a whole number of quantisation groups.
    NotGroupAligned {
        /// Weights in the slice.
        n_weights: u64,
        /// Required group size.
        group: usize,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "io error: {e}"),
            StoreError::BadMagic => write!(f, "not a pmxstore file"),
            StoreError::UnsupportedVersion(v) => write!(f, "unsupported store version {v}"),
            StoreError::BadHeader(m) => write!(f, "bad store header: {m}"),
            StoreError::BadRecord(m) => write!(f, "bad store record: {m}"),
            StoreError::OutOfBounds {
                offset,
                len,
                data_len,
            } => write!(
                f,
                "record spans {offset}..{} but the data section is {data_len} bytes",
                offset.saturating_add(*len)
            ),
            StoreError::Kernel(e) => write!(f, "kernel error: {e}"),
            StoreError::NotGroupAligned { n_weights, group } => write!(
                f,
                "slice of {n_weights} weights is not a multiple of the {group}-weight group"
            ),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

impl From<pmx_kernels::KernelError> for StoreError {
    fn from(e: pmx_kernels::KernelError) -> Self {
        StoreError::Kernel(e)
    }
}

/// A pending slice: its precision, packed bytes, and weight count.
type PendingSlice = (PmxType, Vec<u8>, u64);
/// Key ordering slices by layer, then slot, then kind — which is also the order
/// they are written, so an expert's matrices end up adjacent.
type SliceKey = (u32, u32, u8);

/// Builds a store.
///
/// Slices are added in whatever order is convenient, then [`StoreWriter::finish`]
/// orders them so that an expert's matrices are adjacent and groups fall on
/// alignment boundaries.
#[derive(Debug, Default)]
pub struct StoreWriter {
    slices: BTreeMap<SliceKey, PendingSlice>,
    group_align: u64,
    /// Number of experts per co-activation group. Zero disables grouping.
    group_experts: u32,
}

impl StoreWriter {
    /// A writer with the given group alignment and experts per group.
    pub fn new(group_align: u64, group_experts: u32) -> Self {
        StoreWriter {
            slices: BTreeMap::new(),
            group_align,
            group_experts,
        }
    }

    /// Quantise `weights` to `ty` and add it as a slice.
    pub fn add(
        &mut self,
        layer: u32,
        slot: u32,
        kind: Kind,
        ty: PmxType,
        weights: &[f32],
    ) -> Result<(), StoreError> {
        if weights.len() % GROUP != 0 {
            return Err(StoreError::NotGroupAligned {
                n_weights: weights.len() as u64,
                group: GROUP,
            });
        }
        let mut packed = Vec::new();
        pmx_kernels::pmxq::quantize(ty, weights, &mut packed)?;
        self.slices.insert(
            (layer, slot, kind.tag()),
            (ty, packed, weights.len() as u64),
        );
        Ok(())
    }

    /// Slices added so far.
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Whether nothing has been added.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Write the store to `path`.
    pub fn finish(self, path: impl AsRef<Path>) -> Result<StoreStats, StoreError> {
        // Alignment must not exceed what a group actually holds. Aligning
        // 384 KiB groups to 2 MiB spends more on padding than on weights, which
        // is worse than not grouping at all: the padding is read too.
        let group_payload: u64 = if self.group_experts == 0 {
            0
        } else {
            // One layer's worth: groups never span layers, so summing slot 0
            // across every layer overestimates a group by the layer count and
            // leaves the alignment far too coarse.
            let first_layer = self.slices.keys().next().map(|(l, _, _)| *l).unwrap_or(0);
            let per_expert: u64 = self
                .slices
                .iter()
                .filter(|((l, slot, _), _)| *l == first_layer && *slot == 0)
                .map(|(_, (_, p, _))| p.len() as u64)
                .sum();
            per_expert * u64::from(self.group_experts)
        };
        let effective_align = if self.group_align <= 1 || group_payload == 0 {
            self.group_align
        } else {
            // Round the group payload up to a power of two, floored at 4 KiB so
            // reads stay page-aligned, and never exceed the requested value.
            let want = group_payload.next_power_of_two().max(4096);
            self.group_align.min(want)
        };
        let group_align = effective_align;
        // BTreeMap ordering already gives (layer, slot, kind), which puts an
        // expert's three matrices next to each other — the point being that one
        // read can serve all of them.
        let mut records: Vec<Record> = Vec::with_capacity(self.slices.len());
        let mut payloads: Vec<&[u8]> = Vec::with_capacity(self.slices.len());
        let mut cursor = 0u64;
        let mut padding = 0u64;
        let mut groups = 0usize;

        for (i, ((layer, slot, kind_tag), (ty, packed, n_weights))) in
            self.slices.iter().enumerate()
        {
            let kind = Kind::from_tag(*kind_tag).expect("tags are produced by Kind::tag");
            // Start a new aligned group every `group_experts` experts.
            let starts_group =
                self.group_experts > 0 && *kind_tag == 0 && slot % self.group_experts == 0;
            if group_align > 1 && (starts_group || i == 0) {
                let bumped = align_up(cursor, group_align);
                padding += bumped - cursor;
                cursor = bumped;
                groups += 1;
            }
            records.push(Record {
                layer: *layer,
                slot: *slot,
                kind,
                ty: *ty,
                n_weights: *n_weights,
                offset: cursor,
                nbytes: packed.len() as u64,
            });
            payloads.push(packed);
            cursor += packed.len() as u64;
        }

        let payload_bytes: u64 = payloads.iter().map(|p| p.len() as u64).sum();
        let index_offset = HEADER_BYTES;
        let index_len = (records.len() * RECORD_BYTES) as u64;
        let data_offset = align_up(index_offset + index_len, group_align.max(1));

        let mut w = BufWriter::with_capacity(1 << 20, File::create(path.as_ref())?);
        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(records.len() as u64).to_le_bytes())?;
        w.write_all(&group_align.to_le_bytes())?;
        w.write_all(&index_offset.to_le_bytes())?;
        w.write_all(&data_offset.to_le_bytes())?;
        for r in &records {
            let mut rec = [0u8; RECORD_BYTES];
            rec[0..4].copy_from_slice(&r.layer.to_le_bytes());
            rec[4..8].copy_from_slice(&r.slot.to_le_bytes());
            rec[8..16].copy_from_slice(&r.n_weights.to_le_bytes());
            rec[16..24].copy_from_slice(&r.offset.to_le_bytes());
            rec[24..32].copy_from_slice(&r.nbytes.to_le_bytes());
            rec[32] = r.kind.tag();
            rec[33] = r.ty.tag();
            // rec[34..40] reserved, left zero.
            w.write_all(&rec)?;
        }
        // Pad to the data section.
        let mut written = index_offset + index_len;
        let zeros = vec![0u8; 1 << 16];
        while written < data_offset {
            let n = ((data_offset - written) as usize).min(zeros.len());
            w.write_all(&zeros[..n])?;
            written += n as u64;
        }
        // Payloads, honouring the offsets computed above.
        let mut at = 0u64;
        for (r, p) in records.iter().zip(&payloads) {
            while at < r.offset {
                let n = ((r.offset - at) as usize).min(zeros.len());
                w.write_all(&zeros[..n])?;
                at += n as u64;
            }
            w.write_all(p)?;
            at += p.len() as u64;
        }
        w.flush()?;

        Ok(StoreStats {
            records: records.len(),
            data_bytes: cursor,
            payload_bytes,
            padding_bytes: padding,
            groups,
            index_bytes: index_len,
        })
    }
}

/// Summary of a written store.
#[derive(Debug, Clone, Copy)]
pub struct StoreStats {
    /// Slices written.
    pub records: usize,
    /// Size of the data section, padding included.
    pub data_bytes: u64,
    /// Bytes of actual slice payload, padding excluded.
    ///
    /// Kept separate because conflating the two makes a store with heavy
    /// alignment padding look like it holds more weights than it does.
    pub payload_bytes: u64,
    /// Padding inserted for group alignment.
    pub padding_bytes: u64,
    /// Co-activation groups.
    pub groups: usize,
    /// Index size.
    pub index_bytes: u64,
}

fn align_up(n: u64, align: u64) -> u64 {
    if align <= 1 {
        return n;
    }
    let rem = n % align;
    if rem == 0 {
        n
    } else {
        n + (align - rem)
    }
}

/// A store opened for reading.
#[derive(Debug)]
pub struct Store {
    path: PathBuf,
    records: Vec<Record>,
    /// `by_key[(layer, slot)]` -> indices into `records`.
    by_key: BTreeMap<(u32, u32), Vec<usize>>,
    data_offset: u64,
    group_align: u64,
    file_len: u64,
}

impl Store {
    /// Open and validate a store's index.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path)?;
        let file_len = f.metadata()?.len();
        let mut head = [0u8; HEADER_BYTES as usize];
        f.read_exact(&mut head)?;
        if head[0..8] != MAGIC {
            return Err(StoreError::BadMagic);
        }
        let rd32 =
            |at: usize| u32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
        let rd64 = |at: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&head[at..at + 8]);
            u64::from_le_bytes(b)
        };
        let version = rd32(8);
        if version != VERSION {
            return Err(StoreError::UnsupportedVersion(version));
        }
        let count = rd64(12);
        let group_align = rd64(20);
        let index_offset = rd64(28);
        let data_offset = rd64(36);

        // Bound the index against the real file before allocating for it.
        let index_len = count
            .checked_mul(RECORD_BYTES as u64)
            .ok_or_else(|| StoreError::BadHeader("index length overflows".into()))?;
        if index_offset + index_len > file_len || data_offset > file_len {
            return Err(StoreError::BadHeader(format!(
                "index {index_offset}+{index_len} or data {data_offset} exceeds the {file_len}-byte file"
            )));
        }

        let mut buf = vec![0u8; index_len as usize];
        f.seek(SeekFrom::Start(index_offset))?;
        f.read_exact(&mut buf)?;

        let data_len = file_len - data_offset;
        let mut records = Vec::with_capacity(count as usize);
        let mut by_key: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (i, c) in buf.chunks_exact(RECORD_BYTES).enumerate() {
            let g32 = |at: usize| u32::from_le_bytes([c[at], c[at + 1], c[at + 2], c[at + 3]]);
            let g64 = |at: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&c[at..at + 8]);
                u64::from_le_bytes(b)
            };
            let layer = g32(0);
            let slot = g32(4);
            let n_weights = g64(8);
            let offset = g64(16);
            let nbytes = g64(24);
            let kind = Kind::from_tag(c[32])
                .ok_or_else(|| StoreError::BadRecord(format!("record {i}: bad kind {}", c[32])))?;
            let ty = PmxType::from_tag(c[33])
                .ok_or_else(|| StoreError::BadRecord(format!("record {i}: bad type {}", c[33])))?;

            if n_weights % GROUP as u64 != 0 {
                return Err(StoreError::NotGroupAligned {
                    n_weights,
                    group: GROUP,
                });
            }
            let expect = ty.bytes_for(n_weights as usize)? as u64;
            if expect != nbytes {
                return Err(StoreError::BadRecord(format!(
                    "record {i}: {} weights at {} need {expect} bytes, index says {nbytes}",
                    n_weights,
                    ty.label()
                )));
            }
            if offset
                .checked_add(nbytes)
                .map(|e| e > data_len)
                .unwrap_or(true)
            {
                return Err(StoreError::OutOfBounds {
                    offset,
                    len: nbytes,
                    data_len,
                });
            }
            by_key.entry((layer, slot)).or_default().push(i);
            records.push(Record {
                layer,
                slot,
                kind,
                ty,
                n_weights,
                offset,
                nbytes,
            });
        }

        Ok(Store {
            path,
            records,
            by_key,
            data_offset,
            group_align,
            file_len,
        })
    }

    /// Every record.
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// Group alignment the store was written with.
    pub fn group_align(&self) -> u64 {
        self.group_align
    }

    /// Total file length.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Records making up one expert, in stored order.
    pub fn expert(&self, layer: u32, slot: u32) -> Vec<Record> {
        self.by_key
            .get(&(layer, slot))
            .map(|idx| idx.iter().map(|&i| self.records[i]).collect())
            .unwrap_or_default()
    }

    /// Total bytes one expert occupies.
    pub fn expert_bytes(&self, layer: u32, slot: u32) -> u64 {
        self.expert(layer, slot).iter().map(|r| r.nbytes).sum()
    }

    /// Byte span covering an expert's slices, and whether they are contiguous.
    ///
    /// Contiguity is the property the store is laid out for: if an expert's three
    /// matrices are adjacent, one read serves all of them.
    pub fn expert_span(&self, layer: u32, slot: u32) -> Option<(u64, u64, bool)> {
        let rs = self.expert(layer, slot);
        if rs.is_empty() {
            return None;
        }
        let lo = rs.iter().map(|r| r.offset).min()?;
        let hi = rs.iter().map(|r| r.offset + r.nbytes).max()?;
        let sum: u64 = rs.iter().map(|r| r.nbytes).sum();
        Some((lo, hi - lo, hi - lo == sum))
    }

    /// Read one record's bytes.
    pub fn read_record(&self, r: &Record) -> Result<Vec<u8>, StoreError> {
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_offset + r.offset))?;
        let mut buf = vec![0u8; r.nbytes as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read and dequantise one record.
    pub fn read_weights(&self, r: &Record) -> Result<Vec<f32>, StoreError> {
        let packed = self.read_record(r)?;
        let mut out = Vec::new();
        pmx_kernels::pmxq::dequantize(r.ty, &packed, r.n_weights as usize, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, off: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 / n as f32) * 2.0 - 1.0 + off)
            .collect()
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join("pmx-store-tests");
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    fn build(path: &Path, layers: u32, experts: u32, w: usize) -> StoreStats {
        let mut sw = StoreWriter::new(4096, 4);
        for l in 0..layers {
            for e in 0..experts {
                for k in Kind::ALL {
                    let ty = if e % 2 == 0 { PmxType::Q4 } else { PmxType::Q2 };
                    sw.add(l, e, k, ty, &ramp(w, e as f32 * 0.01)).unwrap();
                }
            }
        }
        sw.finish(path).unwrap()
    }

    #[test]
    fn alignment_never_costs_more_than_it_saves() {
        // Regression guard: a 2 MiB group alignment applied to groups holding a
        // few hundred KiB once produced nine times more padding than payload.
        let p = tmp("padding.pmxstore");
        let mut sw = StoreWriter::new(2 << 20, 8);
        let vals = vec![0.5f32; GROUP * 64];
        for e in 0..32u32 {
            for k in Kind::ALL {
                sw.add(0, e, k, PmxType::Q8, &vals).unwrap();
            }
        }
        let stats = sw.finish(&p).unwrap();
        assert!(
            stats.padding_bytes < stats.payload_bytes,
            "padding {} exceeded payload {}",
            stats.padding_bytes,
            stats.payload_bytes
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn payload_and_padding_are_reported_separately() {
        let p = tmp("report.pmxstore");
        let stats = build(&p, 1, 8, GROUP * 4);
        assert!(stats.payload_bytes > 0);
        assert_eq!(stats.data_bytes, stats.payload_bytes + stats.padding_bytes);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_index_is_exactly_record_bytes_per_record() {
        // Regression guard: the writer once emitted 36 bytes while the reader
        // strode 40, which decoded every record after the first as garbage.
        let p = tmp("recsize.pmxstore");
        let stats = build(&p, 2, 4, GROUP);
        assert_eq!(stats.index_bytes, (stats.records * RECORD_BYTES) as u64);
        let s = Store::open(&p).unwrap();
        assert_eq!(s.records().len(), stats.records);
        // Every record must decode to a sane type and kind.
        for r in s.records() {
            assert!(r.nbytes > 0 && r.n_weights % GROUP as u64 == 0);
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn kind_tags_and_names_round_trip() {
        for k in Kind::ALL {
            assert_eq!(Kind::from_tag(k.tag()), Some(k));
        }
        assert_eq!(Kind::from_tag(9), None);
        assert_eq!(
            Kind::from_tensor_name("blk.3.ffn_up_exps.weight"),
            Some(Kind::Up)
        );
        assert_eq!(Kind::from_tensor_name("blk.3.ffn_gate_inp.weight"), None);
    }

    #[test]
    fn round_trips_through_a_file() {
        let p = tmp("rt.pmxstore");
        let stats = build(&p, 2, 8, GROUP * 4);
        assert_eq!(stats.records, 2 * 8 * 3);

        let s = Store::open(&p).unwrap();
        assert_eq!(s.records().len(), 2 * 8 * 3);
        for l in 0..2u32 {
            for e in 0..8u32 {
                let rs = s.expert(l, e);
                assert_eq!(rs.len(), 3, "layer {l} expert {e}");
                let want = if e % 2 == 0 { PmxType::Q4 } else { PmxType::Q2 };
                assert!(rs.iter().all(|r| r.ty == want));
                let w = s.read_weights(&rs[0]).unwrap();
                assert_eq!(w.len(), GROUP * 4);
            }
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn per_expert_precision_is_actually_honoured() {
        // The reason this format exists: two experts in the same layer stored at
        // different bit widths. GGUF cannot represent this.
        let p = tmp("mixed.pmxstore");
        build(&p, 1, 4, GROUP * 2);
        let s = Store::open(&p).unwrap();
        let even = s.expert_bytes(0, 0);
        let odd = s.expert_bytes(0, 1);
        assert!(
            odd < even,
            "the Q2 expert ({odd} bytes) should be smaller than the Q4 one ({even})"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_experts_matrices_are_stored_contiguously() {
        // One read must be able to serve gate, up and down together.
        let p = tmp("contig.pmxstore");
        build(&p, 1, 8, GROUP * 2);
        let s = Store::open(&p).unwrap();
        for e in 0..8u32 {
            let (_, _, contiguous) = s.expert_span(0, e).unwrap();
            assert!(contiguous, "expert {e} is split across the file");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn groups_land_on_alignment_boundaries() {
        let p = tmp("align.pmxstore");
        build(&p, 1, 8, GROUP * 2);
        let s = Store::open(&p).unwrap();
        let align = s.group_align();
        // Every fourth expert (group_experts = 4) starts a group.
        for e in [0u32, 4] {
            let (lo, _, _) = s.expert_span(0, e).unwrap();
            assert_eq!(lo % align, 0, "expert {e} starts a group at unaligned {lo}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn weights_survive_the_round_trip_within_quantisation_error() {
        let p = tmp("fidelity.pmxstore");
        let src = ramp(GROUP * 8, 0.0);
        let mut sw = StoreWriter::new(4096, 0);
        sw.add(0, 0, Kind::Gate, PmxType::Q8, &src).unwrap();
        sw.finish(&p).unwrap();
        let s = Store::open(&p).unwrap();
        let got = s.read_weights(&s.expert(0, 0)[0]).unwrap();
        assert_eq!(got.len(), src.len());
        let rmse = {
            let mut acc = 0.0f64;
            for (a, b) in src.iter().zip(&got) {
                let d = f64::from(*a) - f64::from(*b);
                acc += d * d;
            }
            (acc / src.len() as f64).sqrt()
        };
        assert!(rmse < 0.01, "round-trip rmse {rmse}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn non_group_aligned_slices_are_refused() {
        let mut sw = StoreWriter::new(4096, 0);
        let bad = vec![0.0f32; GROUP - 1];
        assert!(matches!(
            sw.add(0, 0, Kind::Gate, PmxType::Q4, &bad),
            Err(StoreError::NotGroupAligned { .. })
        ));
    }

    #[test]
    fn a_corrupt_magic_is_rejected() {
        let p = tmp("badmagic.pmxstore");
        build(&p, 1, 2, GROUP);
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
            f.write_all(b"NOTASTOR").unwrap();
        }
        assert!(matches!(Store::open(&p), Err(StoreError::BadMagic)));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_truncated_store_is_rejected_not_read_past() {
        let p = tmp("trunc.pmxstore");
        build(&p, 1, 4, GROUP * 2);
        let len = std::fs::metadata(&p).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        f.set_len(len / 2).unwrap();
        drop(f);
        // Either the header bound check or a record bound check must catch this.
        assert!(Store::open(&p).is_err(), "a halved store was accepted");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_index_declaring_a_wrong_byte_count_is_rejected() {
        // Guards the invariant that ties n_weights, type and nbytes together;
        // without it a crafted index could make a reader allocate or read wrong.
        let p = tmp("badlen.pmxstore");
        build(&p, 1, 2, GROUP);
        let mut bytes = std::fs::read(&p).unwrap();
        // First record's nbytes field sits at HEADER_BYTES + 24.
        let at = HEADER_BYTES as usize + 24;
        bytes[at..at + 8].copy_from_slice(&999_999u64.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();
        assert!(
            Store::open(&p).is_err(),
            "a lying index record was accepted"
        );
        let _ = std::fs::remove_file(&p);
    }
}
