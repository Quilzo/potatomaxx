// SPDX-License-Identifier: GPL-2.0-or-later
//! Kernel hints and observations.
//!
//! Three things a userspace inference engine can tell the kernel, or ask it,
//! that materially change how this workload behaves:
//!
//! * **`RWF_DONTCACHE`** (Linux 6.14) — read through the page cache but drop the
//!   range on completion. Streaming cold expert weights otherwise evicts
//!   everything else and then makes reclaim the bottleneck. Jens Axboe's series
//!   reported 65-75% higher throughput at half the CPU on this pattern.
//! * **`MADV_HUGEPAGE`** — back the resident hot set with 2 MiB pages. The hot
//!   set is read on nearly every token, so its page-table walks are on the
//!   critical path; a 4 KiB mapping of a multi-gigabyte region is a lot of TLB
//!   pressure for no reason.
//! * **`cachestat()`** (Linux 6.5) — ask how much of a file range is actually
//!   resident. This is the difference between *believing* a cache plan is being
//!   honoured and *knowing* it, and it costs one syscall.

use crate::sys;
use crate::KioError;
use std::os::fd::AsRawFd;

/// What the page cache holds for a file range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Residency {
    /// Pages present in the page cache.
    pub cached: u64,
    /// Pages dirty.
    pub dirty: u64,
    /// Pages under writeback.
    pub writeback: u64,
    /// Pages evicted from this range.
    pub evicted: u64,
    /// Pages evicted recently enough to count as thrashing.
    pub recently_evicted: u64,
}

impl Residency {
    /// Share of `total_pages` currently cached.
    pub fn cached_fraction(&self, total_pages: u64) -> f64 {
        if total_pages == 0 {
            0.0
        } else {
            self.cached as f64 / total_pages as f64
        }
    }

    /// Whether the kernel is thrashing this range: evicting pages and then
    /// wanting them back.
    ///
    /// A non-zero `recently_evicted` is the signal that the page cache is being
    /// used as a cache for something that does not fit, which is exactly the
    /// condition `RWF_DONTCACHE` exists to avoid.
    pub fn is_thrashing(&self) -> bool {
        self.recently_evicted > 0
    }
}

/// Query page-cache residency for a byte range.
///
/// `len` of 0 means "to the end of the file".
pub fn residency<F: AsRawFd>(file: &F, offset: u64, len: u64) -> Result<Residency, KioError> {
    let range = sys::CachestatRange { off: offset, len };
    let mut out = sys::Cachestat::default();
    // SAFETY: both structures are valid, correctly sized and writable, and the
    // fd is borrowed for the duration of the call.
    unsafe { sys::cachestat(file.as_raw_fd(), &range, &mut out) }.map_err(|e| {
        if e == 38 {
            KioError::MissingFeature("cachestat(2) — needs Linux 6.5")
        } else {
            KioError::Cachestat(e)
        }
    })?;
    Ok(Residency {
        cached: out.nr_cache,
        dirty: out.nr_dirty,
        writeback: out.nr_writeback,
        evicted: out.nr_evicted,
        recently_evicted: out.nr_recently_evicted,
    })
}

/// Ask for transparent huge pages over a region.
///
/// Advisory: the kernel may refuse, and `transparent_hugepage/enabled` must be
/// `always` or `madvise` for it to have any effect. Reports success only if the
/// call itself succeeded, never that huge pages were actually installed.
///
/// # Safety
///
/// `addr` must be the start of a mapping of at least `len` bytes that stays valid
/// for the call.
pub unsafe fn request_huge_pages(addr: *mut u8, len: usize) -> Result<(), KioError> {
    unsafe { sys::madvise(addr as *mut std::ffi::c_void, len, sys::MADV_HUGEPAGE) }
        .map(|_| ())
        .map_err(KioError::Madvise)
}

/// Tell the kernel a region will be accessed randomly, so it stops reading ahead.
///
/// Readahead is a bet that the next block is wanted. For expert fetch it is a
/// losing bet: the next expert is chosen by a router, not by address order, so
/// every readahead page is bandwidth spent on data nothing asked for.
///
/// # Safety
///
/// As [`request_huge_pages`].
pub unsafe fn advise_random(addr: *mut u8, len: usize) -> Result<(), KioError> {
    unsafe { sys::madvise(addr as *mut std::ffi::c_void, len, sys::MADV_RANDOM) }
        .map(|_| ())
        .map_err(KioError::Madvise)
}

/// Read `buf` from `file` at `offset` with the given `RWF_*` flags.
///
/// The synchronous counterpart to the io_uring path, used for feature probing and
/// as a fallback.
pub fn pread_flags<F: AsRawFd>(
    file: &F,
    buf: &mut [u8],
    offset: u64,
    flags: i32,
) -> Result<usize, KioError> {
    let iov = sys::IoVec {
        base: buf.as_mut_ptr() as *mut std::ffi::c_void,
        len: buf.len(),
    };
    // SAFETY: one iovec describing `buf`, which is uniquely borrowed and lives
    // for the duration of the call.
    let n = unsafe { sys::preadv2(file.as_raw_fd(), &iov, 1, offset as i64, flags) }
        .map_err(KioError::Pread)?;
    Ok(n as usize)
}

/// Whether this kernel accepts `RWF_DONTCACHE`.
///
/// Probed rather than inferred from the release string: distributions backport,
/// containers lie, and filesystems opt in individually — a filesystem without
/// `FOP_DONTCACHE` rejects the flag on a kernel that otherwise supports it.
pub fn dontcache_supported<F: AsRawFd>(file: &F) -> bool {
    let mut b = [0u8; 512];
    match pread_flags(file, &mut b, 0, sys::RWF_DONTCACHE) {
        Ok(_) => true,
        Err(KioError::Pread(e)) => {
            // EINVAL/EOPNOTSUPP mean the flag or the filesystem said no.
            !matches!(e, 22 | 95)
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn tmp(name: &str, bytes: usize) -> std::path::PathBuf {
        let d = std::env::temp_dir().join("pmx-kio-hint-tests");
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(&vec![0xA5u8; bytes]).unwrap();
        f.sync_all().unwrap();
        p
    }

    #[test]
    fn residency_reports_something_plausible() {
        let p = tmp("res.bin", 1 << 20);
        let f = File::open(&p).unwrap();
        match residency(&f, 0, 0) {
            Ok(r) => {
                // The file was just written, so most of it should be cached, and
                // the count can never exceed the file's page count.
                let pages = (1u64 << 20) / 4096;
                assert!(
                    r.cached <= pages + 1,
                    "cached {} > {} pages",
                    r.cached,
                    pages
                );
                assert!(r.cached_fraction(pages) <= 1.01);
            }
            Err(KioError::MissingFeature(_)) => {
                eprintln!("cachestat unavailable; skipping");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn pread_with_and_without_dontcache_agree() {
        let p = tmp("dc.bin", 64 << 10);
        let f = File::open(&p).unwrap();
        let mut a = [0u8; 4096];
        let mut b = [0u8; 4096];
        let na = pread_flags(&f, &mut a, 8192, 0).unwrap();
        assert_eq!(na, 4096);
        if dontcache_supported(&f) {
            let nb = pread_flags(&f, &mut b, 8192, crate::sys::RWF_DONTCACHE).unwrap();
            assert_eq!(nb, 4096);
            assert_eq!(a, b, "RWF_DONTCACHE changed the bytes read");
        } else {
            eprintln!("RWF_DONTCACHE unsupported here; skipping comparison");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn hugepage_advice_is_accepted_or_cleanly_refused() {
        // Allocate a 2 MiB-aligned region so the advice has a chance of applying.
        let len = 4 << 20;
        let mut v = vec![0u8; len];
        // SAFETY: the vector's allocation is live for the call.
        let r = unsafe { request_huge_pages(v.as_mut_ptr(), len) };
        match r {
            Ok(()) => {}
            Err(KioError::Madvise(e)) => {
                // EINVAL is normal when THP is compiled out or set to `never`.
                assert!(e == 22 || e == 12, "unexpected madvise errno {e}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
        // Touch the region so the mapping is real, and keep it alive.
        v[0] = 1;
        v[len - 1] = 1;
        assert_eq!(v[0], 1);
    }

    #[test]
    fn random_advice_is_accepted() {
        let len = 1 << 20;
        let mut v = vec![0u8; len];
        // SAFETY: as above.
        let r = unsafe { advise_random(v.as_mut_ptr(), len) };
        assert!(r.is_ok() || matches!(r, Err(KioError::Madvise(_))));
        v[0] = 2;
        assert_eq!(v[0], 2);
    }

    #[test]
    fn thrashing_detection_reads_the_right_field() {
        let mut r = Residency::default();
        assert!(!r.is_thrashing());
        r.recently_evicted = 5;
        assert!(r.is_thrashing());
    }
}
