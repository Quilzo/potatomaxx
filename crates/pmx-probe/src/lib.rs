// SPDX-License-Identifier: GPL-2.0-or-later
//! Measuring the storage bandwidth surface a layout plan is optimised against.
//!
//! Every decision `potatomaxx` makes rests on one empirical fact: the read
//! bandwidth of a storage device is a strong function of *request size* and
//! *how many requests are in flight*, and the spread between the corners is
//! enormous. On the development machine (an i5-1235U laptop, NVMe behind WSL2)
//! the surface looks like this:
//!
//! ```text
//!      blob      QD1      QD4      QD8     QD16
//!    64 KiB     0.12     0.41     0.82     1.36  GB/s
//!   256 KiB     0.26     1.24     2.15     2.02
//!     1 MiB     0.64     1.70     2.21     2.40
//!     2 MiB     0.79     2.10     2.41     2.34
//!     8 MiB     1.13     1.72     2.10     2.06
//!    32 MiB     1.83     1.99     1.77     1.92
//! ```
//!
//! Two facts drive the whole design. Queue depth is worth about 3x at a fixed
//! request size, and request size is worth about 20x between the worst and best
//! corners. A demand-paged `mmap` sits in the top-left cell; that is why
//! engines which fall back to paging collapse on memory-constrained machines.
//!
//! Because this surface is device-specific, it is measured rather than assumed.
//! [`measure`] produces one, and [`Surface::bandwidth_at`] interpolates it so a
//! partitioning objective can be scored in bytes-per-second rather than in
//! abstract group counts.

#![warn(missing_docs)]
// This crate is one of the two places in the workspace permitted to use
// `unsafe`, and the only one in this build. It needs page-aligned buffers for
// `O_DIRECT`, which safe Rust cannot express. Every unsafe block below carries
// its invariant, and `AlignedBuf` is covered by a test asserting alignment.
// Everything outside `wl-probe` and the SIMD kernels stays in safe Rust.
#![allow(unsafe_code)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `O_DIRECT` on Linux. Bypasses the page cache so the measurement reflects the
/// device, not data the kernel already had in RAM.
///
/// There is no portable equivalent. macOS needs `fcntl(F_NOCACHE)`, which would
/// mean a `libc` dependency this workspace does not take. On platforms without
/// cache bypass the probe still runs, but reports *cached* bandwidth, which can
/// be several times the device's real figure. [`Surface::cache_bypassed`] records
/// which happened so a surface is never mistaken for something it is not.
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = 0x4000;

/// Whether this build can bypass the page cache while probing.
pub const CACHE_BYPASS_AVAILABLE: bool = cfg!(target_os = "linux");

/// Alignment required for `O_DIRECT` buffers and offsets.
pub const DIRECT_ALIGN: usize = 4096;

/// A page-aligned heap buffer, required for `O_DIRECT` reads.
///
/// # Safety invariants
///
/// `ptr` is the exact allocation returned for `layout`, is non-null, and is
/// freed exactly once in `Drop` with that same layout. `len` never exceeds the
/// allocated size, so the slice handed out by [`AlignedBuf::as_mut`] always
/// stays inside the allocation.
struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the buffer is a plain owned byte allocation with no interior
// references and no thread affinity; moving it between threads is sound.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> std::io::Result<Self> {
        let layout = Layout::from_size_align(len, DIRECT_ALIGN)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        // SAFETY: `layout` has non-zero size (callers pass len >= DIRECT_ALIGN)
        // and a valid power-of-two alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "aligned allocation failed",
            ));
        }
        Ok(AlignedBuf { ptr, layout })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is a live allocation of exactly `layout.size()` bytes,
        // uniquely borrowed for the lifetime of the returned slice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with this exact `layout` and is
        // freed only here, once.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

/// One measured cell of the bandwidth surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    /// Request size in bytes.
    pub blob_bytes: u64,
    /// Number of concurrent readers.
    pub queue_depth: usize,
    /// Achieved read bandwidth in bytes per second.
    pub bytes_per_sec: f64,
}

/// A measured device bandwidth surface.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    /// Measured cells.
    pub cells: Vec<Cell>,
    /// Free-text note recording what was measured, for provenance.
    pub note: String,
    /// Whether the page cache was bypassed. When false, the figures are inflated
    /// by whatever the kernel had cached and must not be read as device speed.
    pub cache_bypassed: bool,
}

impl Surface {
    /// Best bandwidth anywhere on the surface.
    pub fn peak(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| c.bytes_per_sec)
            .fold(0.0, f64::max)
    }

    /// Estimated bandwidth for a given request size and queue depth.
    ///
    /// Picks the measured cell closest in log-space on both axes. This is
    /// deliberately crude: the surface is flat enough near its plateau that
    /// nearest-neighbour is within measurement noise, and pretending to a
    /// smooth interpolation would overstate what the data supports.
    pub fn bandwidth_at(&self, blob_bytes: u64, queue_depth: usize) -> Option<f64> {
        if self.cells.is_empty() {
            return None;
        }
        let lb = (blob_bytes.max(1) as f64).ln();
        let lq = (queue_depth.max(1) as f64).ln();
        let mut best: Option<(f64, f64)> = None;
        for c in &self.cells {
            let d = ((c.blob_bytes.max(1) as f64).ln() - lb).abs()
                + ((c.queue_depth.max(1) as f64).ln() - lq).abs();
            let better = match best {
                None => true,
                Some((bd, _)) => d < bd,
            };
            if better {
                best = Some((d, c.bytes_per_sec));
            }
        }
        best.map(|(_, bw)| bw)
    }

    /// Serialise to a small self-describing JSON document.
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n  \"note\": \"");
        for ch in self.note.chars() {
            match ch {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                '\n' => s.push_str("\\n"),
                c => s.push(c),
            }
        }
        s.push_str(&format!(
            "\",\n  \"cache_bypassed\": {},\n  \"cells\": [\n",
            self.cache_bypassed
        ));
        for (i, c) in self.cells.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"blob_bytes\": {}, \"queue_depth\": {}, \"bytes_per_sec\": {:.0}}}",
                c.blob_bytes, c.queue_depth, c.bytes_per_sec
            ));
            if i + 1 < self.cells.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ]\n}\n");
        s
    }

    /// Parse the document produced by [`Surface::to_json`].
    ///
    /// A deliberately narrow reader for our own output — not a general JSON
    /// parser. Unrecognised input yields `None`.
    pub fn from_json(text: &str) -> Option<Self> {
        /// Read the number following `key` within `chunk`.
        fn field(chunk: &str, key: &str) -> Option<f64> {
            let at = chunk.find(key)? + key.len();
            let rest = chunk[at..].trim_start();
            let end = rest
                .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        }

        let mut cells = Vec::new();
        for chunk in text.split('{').skip(1) {
            let chunk = match chunk.find('}') {
                Some(i) => &chunk[..i],
                None => continue,
            };
            let blob = field(chunk, "\"blob_bytes\":");
            let qd = field(chunk, "\"queue_depth\":");
            let bw = field(chunk, "\"bytes_per_sec\":");
            if let (Some(blob), Some(qd), Some(bw)) = (blob, qd, bw) {
                cells.push(Cell {
                    blob_bytes: blob as u64,
                    queue_depth: qd as usize,
                    bytes_per_sec: bw,
                });
            }
        }
        if cells.is_empty() {
            return None;
        }
        let note = text
            .find("\"note\":")
            .and_then(|i| {
                let rest = &text[i + 7..];
                let a = rest.find('"')? + 1;
                let b = rest[a..].find('"')? + a;
                Some(rest[a..b].to_string())
            })
            .unwrap_or_default();
        let cache_bypassed = !text.contains("\"cache_bypassed\": false");
        Some(Surface {
            cells,
            note,
            cache_bypassed,
        })
    }
}

/// How much of the surface to measure.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Directory the scratch corpus is written into.
    pub dir: PathBuf,
    /// Corpus size. Must comfortably exceed RAM-cached reuse to be meaningful.
    pub corpus_bytes: u64,
    /// Request sizes to sweep.
    pub blob_sizes: Vec<u64>,
    /// Queue depths to sweep.
    pub queue_depths: Vec<usize>,
    /// Traffic per measured cell, as a ceiling.
    pub traffic_per_cell: u64,
    /// Wall-clock budget per cell, in milliseconds.
    ///
    /// A fixed byte budget is the wrong knob: 600 MiB costs half a second in the
    /// fast corner of the surface and half a minute in the slow one, so sweeping
    /// small requests would dominate the whole run. Each cell instead reads
    /// until whichever limit it hits first, and bandwidth is computed from the
    /// bytes actually moved.
    pub ms_per_cell: u64,
    /// Minimum reads per worker, so a very slow cell still yields a real figure.
    pub min_reads_per_worker: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig {
            dir: PathBuf::from("."),
            corpus_bytes: 2 << 30,
            // Sweep down to 4 KiB (the O_DIRECT alignment floor). Fine-grained
            // MoE checkpoints have expert slices well under 64 KiB, and that is
            // precisely where request size matters most, so a surface that
            // starts at 64 KiB cannot resolve the layouts we care about.
            blob_sizes: vec![
                4 << 10,
                16 << 10,
                32 << 10,
                64 << 10,
                256 << 10,
                1 << 20,
                2 << 20,
                8 << 20,
                32 << 20,
            ],
            queue_depths: vec![1, 4, 8, 16],
            traffic_per_cell: 600 << 20,
            ms_per_cell: 400,
            min_reads_per_worker: 4,
        }
    }
}

/// Measure the bandwidth surface of the device holding `cfg.dir`.
///
/// Writes a scratch corpus, sweeps request size against queue depth with
/// `O_DIRECT` random reads, then removes the corpus. Reported figures are the
/// device's, not the page cache's.
pub fn measure(cfg: &ProbeConfig) -> std::io::Result<Surface> {
    let path = cfg.dir.join("pmx-probe-corpus.bin");
    write_corpus(&path, cfg.corpus_bytes)?;
    let mut cells = Vec::new();
    for &blob in &cfg.blob_sizes {
        if blob < DIRECT_ALIGN as u64 || blob % DIRECT_ALIGN as u64 != 0 {
            continue;
        }
        if blob > cfg.corpus_bytes / 4 {
            continue;
        }
        for &qd in &cfg.queue_depths {
            let bw = bench_cell(&path, cfg.corpus_bytes, blob, qd, cfg)?;
            cells.push(Cell {
                blob_bytes: blob,
                queue_depth: qd,
                bytes_per_sec: bw,
            });
        }
    }
    let _ = std::fs::remove_file(&path);
    Ok(Surface {
        cells,
        note: format!(
            "{} random reads over a {} MiB corpus in {}",
            if CACHE_BYPASS_AVAILABLE {
                "O_DIRECT"
            } else {
                "PAGE-CACHED (no O_DIRECT on this platform)"
            },
            cfg.corpus_bytes >> 20,
            cfg.dir.display()
        ),
        cache_bypassed: CACHE_BYPASS_AVAILABLE,
    })
}

fn write_corpus(path: &Path, bytes: u64) -> std::io::Result<()> {
    if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) == bytes {
        return Ok(());
    }
    let mut f = File::create(path)?;
    // A non-trivial pattern, so a device that dedupes or compresses cannot
    // flatter itself with a file of zeroes.
    let mut chunk = vec![0u8; 8 << 20];
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for b in chunk.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = (x >> 24) as u8;
    }
    let mut left = bytes;
    while left > 0 {
        let n = left.min(chunk.len() as u64) as usize;
        f.write_all(&chunk[..n])?;
        left -= n as u64;
    }
    f.sync_all()
}

fn bench_cell(
    path: &Path,
    corpus: u64,
    blob: u64,
    qd: usize,
    cfg: &ProbeConfig,
) -> std::io::Result<f64> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    #[cfg(target_os = "linux")]
    opts.custom_flags(O_DIRECT);
    let f = Arc::new(opts.open(path)?);

    let slots = (corpus / blob).max(1);
    let cap = ((cfg.traffic_per_cell / blob).max(qd as u64) / qd as u64).max(1);
    let budget = Duration::from_millis(cfg.ms_per_cell);
    let min_reads = cfg.min_reads_per_worker.max(1);

    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(qd);
    for t in 0..qd {
        let f = Arc::clone(&f);
        handles.push(std::thread::spawn(move || -> std::io::Result<u64> {
            let mut buf = AlignedBuf::new(blob as usize)?;
            let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut done = 0u64;
            let start = Instant::now();
            while done < cap {
                // Check the clock every few reads rather than every read, so
                // timing overhead does not distort the small-request cells.
                if done >= min_reads && done % 8 == 0 && start.elapsed() >= budget {
                    break;
                }
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let off = (x >> 16) % slots * blob;
                f.read_exact_at(buf.as_mut(), off)?;
                done += 1;
            }
            Ok(done)
        }));
    }
    let mut total_reads = 0u64;
    for h in handles {
        total_reads += h
            .join()
            .map_err(|_| std::io::Error::other("probe worker panicked"))??;
    }
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    Ok((total_reads * blob) as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surf() -> Surface {
        Surface {
            note: "test".into(),
            cache_bypassed: true,
            cells: vec![
                Cell {
                    blob_bytes: 64 << 10,
                    queue_depth: 1,
                    bytes_per_sec: 0.12e9,
                },
                Cell {
                    blob_bytes: 2 << 20,
                    queue_depth: 1,
                    bytes_per_sec: 0.79e9,
                },
                Cell {
                    blob_bytes: 2 << 20,
                    queue_depth: 8,
                    bytes_per_sec: 2.41e9,
                },
            ],
        }
    }

    #[test]
    fn nearest_cell_lookup() {
        let s = surf();
        // Exact hits.
        assert_eq!(s.bandwidth_at(2 << 20, 8), Some(2.41e9));
        assert_eq!(s.bandwidth_at(64 << 10, 1), Some(0.12e9));
        // A 1 MiB request at QD8 is nearest the 2 MiB/QD8 cell.
        assert_eq!(s.bandwidth_at(1 << 20, 8), Some(2.41e9));
    }

    #[test]
    fn peak_is_the_best_cell() {
        assert_eq!(surf().peak(), 2.41e9);
    }

    #[test]
    fn json_round_trip() {
        let a = surf();
        let b = Surface::from_json(&a.to_json()).expect("parses");
        assert_eq!(a.cells.len(), b.cells.len());
        for (x, y) in a.cells.iter().zip(&b.cells) {
            assert_eq!(x.blob_bytes, y.blob_bytes);
            assert_eq!(x.queue_depth, y.queue_depth);
            assert!((x.bytes_per_sec - y.bytes_per_sec).abs() < 1.0);
        }
        assert_eq!(b.note, "test");
        assert!(b.cache_bypassed);
    }

    #[test]
    fn a_cached_surface_round_trips_as_cached() {
        let mut a = surf();
        a.cache_bypassed = false;
        let b = Surface::from_json(&a.to_json()).expect("parses");
        assert!(
            !b.cache_bypassed,
            "a page-cached surface must not come back claiming otherwise"
        );
    }

    #[test]
    fn empty_surface_has_no_estimate() {
        assert_eq!(Surface::default().bandwidth_at(1 << 20, 8), None);
    }

    #[test]
    fn aligned_buffer_is_page_aligned() {
        let mut b = AlignedBuf::new(8192).unwrap();
        assert_eq!(b.as_mut().len(), 8192);
        assert_eq!(b.ptr as usize % DIRECT_ALIGN, 0);
    }
}
