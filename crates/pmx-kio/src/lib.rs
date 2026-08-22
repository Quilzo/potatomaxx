// SPDX-License-Identifier: GPL-2.0-or-later
//! Kernel I/O paths for expert streaming.
//!
//! # What belongs in the kernel, and what does not
//!
//! It is worth being blunt, because the temptation runs the other way: **an
//! inference engine cannot go in the Linux kernel, and should not.** Two reasons,
//! neither of them a matter of taste.
//!
//! Floating point is prohibited in kernel code. The escape hatch,
//! `kernel_fpu_begin()`/`kernel_fpu_end()`, exists for short crypto and RAID
//! kernels and disables preemption for its duration — the documentation says the
//! critical section "should be minimized". A GEMM is the opposite of minimal, so
//! running one there would create unbounded non-preemptible sections and destroy
//! the latency guarantees the whole system depends on.
//!
//! And a model is *policy*. "Mechanism in the kernel, policy in userspace" is the
//! oldest rule in the project, and the kernel's own answer to "we need
//! application-specific policy" is now BPF: `sched_ext` for scheduling (merged in
//! 6.12), and research frameworks like `cache_ext` and FetchBPF for page-cache
//! and prefetch policy. Even IBM's ML-LIB RFC, a far more modest proposal than an
//! inference engine, keeps the models in userspace and puts only *proxies* in the
//! kernel.
//!
//! So this crate does the honest thing: it uses the kernel interfaces that exist,
//! properly, and measures what they buy.
//!
//! | mechanism | since | what it is for |
//! |---|---|---|
//! | io_uring batched reads | 5.1 | real queue depth, one syscall per batch |
//! | `RWF_DONTCACHE` | 6.14 | stream cold weights without evicting everything |
//! | `MADV_HUGEPAGE` | long-standing | fewer TLB misses on the resident hot set |
//! | `MADV_RANDOM` | long-standing | stop readahead betting on the wrong block |
//! | `cachestat(2)` | 6.5 | verify residency instead of assuming it |
//!
//! Every one is probed at runtime rather than inferred from a version string:
//! distributions backport, containers lie, and `RWF_DONTCACHE` additionally
//! requires the filesystem to have opted in via `FOP_DONTCACHE`.

#![warn(missing_docs)]
// This crate is the third permitted `unsafe`, for raw syscalls and the shared
// io_uring mappings. Safe Rust cannot express either. Every unsafe block carries
// its invariant, and the correctness of the ring is pinned by a test that
// compares batched reads byte-for-byte against ordinary `std` reads.
#![allow(unsafe_code)]

pub mod hints;
pub mod sys;
pub mod uring;

pub use hints::{residency, Residency};
pub use uring::{ReadOp, ReadResult, Ring, StreamResult};

use std::fmt;

/// Anything that can go wrong in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KioError {
    /// `io_uring_setup(2)` failed with this errno.
    UringSetup(i32),
    /// Mapping a ring failed with this errno.
    UringMmap(i32),
    /// `io_uring_enter(2)` failed with this errno.
    UringEnter(i32),
    /// `cachestat(2)` failed with this errno.
    Cachestat(i32),
    /// `madvise(2)` failed with this errno.
    Madvise(i32),
    /// `preadv2(2)` failed with this errno.
    Pread(i32),
    /// A kernel or filesystem feature this path needs is absent.
    MissingFeature(&'static str),
    /// More operations than the ring has submission slots.
    BatchTooLarge {
        /// Operations requested.
        asked: usize,
        /// Slots available.
        limit: usize,
    },
    /// An operation referenced a buffer slot that does not exist.
    BadSlot(u32),
    /// A destination buffer is smaller than its read.
    BufferTooSmall {
        /// Which slot.
        slot: u32,
        /// Bytes needed.
        need: usize,
        /// Bytes available.
        have: usize,
    },
}

impl fmt::Display for KioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = |n: &i32| std::io::Error::from_raw_os_error(*n);
        match self {
            KioError::UringSetup(n) => write!(f, "io_uring_setup failed: {}", e(n)),
            KioError::UringMmap(n) => write!(f, "mapping the io_uring failed: {}", e(n)),
            KioError::UringEnter(n) => write!(f, "io_uring_enter failed: {}", e(n)),
            KioError::Cachestat(n) => write!(f, "cachestat failed: {}", e(n)),
            KioError::Madvise(n) => write!(f, "madvise failed: {}", e(n)),
            KioError::Pread(n) => write!(f, "preadv2 failed: {}", e(n)),
            KioError::MissingFeature(w) => write!(f, "kernel feature unavailable: {w}"),
            KioError::BatchTooLarge { asked, limit } => {
                write!(
                    f,
                    "batch of {asked} exceeds the ring's {limit} submission slots"
                )
            }
            KioError::BadSlot(s) => write!(f, "operation references unknown buffer slot {s}"),
            KioError::BufferTooSmall { slot, need, have } => {
                write!(
                    f,
                    "buffer for slot {slot} holds {have} bytes but the read needs {need}"
                )
            }
        }
    }
}

impl std::error::Error for KioError {}

/// Which kernel facilities are usable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// io_uring can be set up.
    pub io_uring: bool,
    /// The kernel and filesystem accept `RWF_DONTCACHE`.
    pub dontcache: bool,
    /// `cachestat(2)` is present.
    pub cachestat: bool,
    /// Transparent huge pages are enabled in some mode.
    pub thp: bool,
}

impl Capabilities {
    /// Probe the running kernel.
    ///
    /// `probe_file` is opened read-only and used for the flag and syscall
    /// probes; it must exist. Probing beats parsing `uname`.
    pub fn probe(probe_file: &std::path::Path) -> Capabilities {
        let f = std::fs::File::open(probe_file).ok();
        let dontcache = f.as_ref().map(hints::dontcache_supported).unwrap_or(false);
        let cachestat = f
            .as_ref()
            .map(|h| !matches!(hints::residency(h, 0, 0), Err(KioError::MissingFeature(_))))
            .unwrap_or(false);
        let thp = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
            .map(|s| !s.contains("[never]"))
            .unwrap_or(false);
        Capabilities {
            io_uring: uring::available(),
            dontcache,
            cachestat,
            thp,
        }
    }

    /// A one-line summary for reports.
    pub fn summary(&self) -> String {
        let m = |b: bool| if b { "yes" } else { "no" };
        format!(
            "io_uring={} RWF_DONTCACHE={} cachestat={} THP={}",
            m(self.io_uring),
            m(self.dontcache),
            m(self.cachestat),
            m(self.thp)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_probe_does_not_panic_and_reports() {
        let c = Capabilities::probe(std::path::Path::new("/etc/hostname"));
        let s = c.summary();
        assert!(s.contains("io_uring="));
        assert!(s.contains("RWF_DONTCACHE="));
        // Print for the record; different CI kernels will differ.
        eprintln!("capabilities: {s}");
    }

    #[test]
    fn errors_render_errnos_as_messages() {
        let s = format!("{}", KioError::UringSetup(38));
        assert!(s.contains("io_uring_setup"), "{s}");
        let s = format!("{}", KioError::MissingFeature("cachestat(2)"));
        assert!(s.contains("cachestat"), "{s}");
    }
}
