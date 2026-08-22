// SPDX-License-Identifier: GPL-2.0-or-later
//! Raw syscall bindings.
//!
//! Written by hand rather than pulled from `libc` so the workspace keeps its
//! zero-dependency property. Only the handful of numbers and structures this
//! project needs are defined, and each is annotated with the kernel version that
//! introduced it so a runtime probe can be matched against a documented
//! requirement rather than a guess.

#![allow(missing_docs)]

use std::ffi::c_void;

// Syscall numbers are per-ABI, so they must be gated on the OS as well as the
// architecture. Gating on architecture alone once meant a macOS aarch64 build
// compiled with Linux numbers and actually issued one; the runner reported
// SIGSYS. Calling a wrong syscall is far worse than reporting a missing feature,
// so on any non-Linux target every wrapper below fails closed with ENOSYS and no
// syscall instruction is emitted at all.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod nr {
    /// `io_uring_setup(2)` — Linux 5.1.
    pub const IO_URING_SETUP: i64 = 425;
    /// `io_uring_enter(2)` — Linux 5.1.
    pub const IO_URING_ENTER: i64 = 426;
    /// `io_uring_register(2)` — Linux 5.1.
    pub const IO_URING_REGISTER: i64 = 427;
    /// `cachestat(2)` — Linux 6.5.
    pub const CACHESTAT: i64 = 451;
    pub const MADVISE: i64 = 28;
    pub const MMAP: i64 = 9;
    pub const MUNMAP: i64 = 11;
    pub const PREADV2: i64 = 327;
    pub const CLOSE: i64 = 3;
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub mod nr {
    pub const IO_URING_SETUP: i64 = 425;
    pub const IO_URING_ENTER: i64 = 426;
    pub const IO_URING_REGISTER: i64 = 427;
    pub const CACHESTAT: i64 = 451;
    pub const MADVISE: i64 = 233;
    pub const MMAP: i64 = 222;
    pub const MUNMAP: i64 = 215;
    pub const PREADV2: i64 = 286;
    pub const CLOSE: i64 = 57;
}

/// `RWF_DONTCACHE` — Linux 6.14.
///
/// Reads through the page cache but drops the range once the I/O completes.
/// Streaming tens of gigabytes of cold expert weights through the page cache
/// evicts everything else and then makes reclaim the bottleneck; this asks the
/// kernel not to retain what will not be read again.
pub const RWF_DONTCACHE: i32 = 0x0000_0080;

pub const MADV_HUGEPAGE: i32 = 14;
pub const MADV_NOHUGEPAGE: i32 = 15;
pub const MADV_WILLNEED: i32 = 3;
pub const MADV_DONTNEED: i32 = 4;
pub const MADV_RANDOM: i32 = 1;

pub const PROT_READ: i32 = 0x1;
pub const PROT_WRITE: i32 = 0x2;
pub const MAP_SHARED: i32 = 0x01;
pub const MAP_POPULATE: i32 = 0x8000;
pub const MAP_FAILED: isize = -1;

// io_uring flags and offsets.
pub const IORING_OFF_SQ_RING: u64 = 0;
pub const IORING_OFF_CQ_RING: u64 = 0x0800_0000;
pub const IORING_OFF_SQES: u64 = 0x1000_0000;

pub const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
pub const IORING_FEAT_NODROP: u32 = 1 << 1;
pub const IORING_FEAT_SUBMIT_STABLE: u32 = 1 << 2;

pub const IORING_ENTER_GETEVENTS: u32 = 1 << 0;

pub const IORING_OP_READ: u8 = 22;

/// `struct io_sqring_offsets`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SqRingOffsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub flags: u32,
    pub dropped: u32,
    pub array: u32,
    pub resv1: u32,
    pub user_addr: u64,
}

/// `struct io_cqring_offsets`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CqRingOffsets {
    pub head: u32,
    pub tail: u32,
    pub ring_mask: u32,
    pub ring_entries: u32,
    pub overflow: u32,
    pub cqes: u32,
    pub flags: u32,
    pub resv1: u32,
    pub user_addr: u64,
}

/// `struct io_uring_params`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IoUringParams {
    pub sq_entries: u32,
    pub cq_entries: u32,
    pub flags: u32,
    pub sq_thread_cpu: u32,
    pub sq_thread_idle: u32,
    pub features: u32,
    pub wq_fd: u32,
    pub resv: [u32; 3],
    pub sq_off: SqRingOffsets,
    pub cq_off: CqRingOffsets,
}

/// `struct io_uring_sqe`, the 64-byte submission entry.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IoUringSqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    pub off: u64,
    pub addr: u64,
    pub len: u32,
    /// Union in the kernel; we only ever use it as `rw_flags`, which is where
    /// `RWF_DONTCACHE` goes.
    pub rw_flags: i32,
    pub user_data: u64,
    pub buf_index: u16,
    pub personality: u16,
    pub splice_fd_in: i32,
    pub addr3: u64,
    pub pad2: u64,
}

/// `struct io_uring_cqe`, the 16-byte completion entry.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IoUringCqe {
    pub user_data: u64,
    /// Bytes transferred, or a negative errno.
    pub res: i32,
    pub flags: u32,
}

/// `struct cachestat_range`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CachestatRange {
    pub off: u64,
    pub len: u64,
}

/// `struct cachestat`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Cachestat {
    pub nr_cache: u64,
    pub nr_dirty: u64,
    pub nr_writeback: u64,
    pub nr_evicted: u64,
    pub nr_recently_evicted: u64,
}

/// `struct iovec`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IoVec {
    pub base: *mut c_void,
    pub len: usize,
}

/// Result of a raw syscall: `Ok(value)` or `Err(errno)`.
pub type SysResult = Result<i64, i32>;

/// `ENOSYS`, returned by every wrapper on targets this crate does not support.
pub const ENOSYS: i32 = 38;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;

    unsafe extern "C" {
        fn syscall(num: i64, ...) -> i64;
    }

    fn wrap(ret: i64) -> SysResult {
        if ret < 0 {
            // The libc wrapper returns -1 and sets errno, but on some paths the
            // raw value is the negated errno. Handle both.
            let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            Err(if e != 0 { e } else { (-ret) as i32 })
        } else {
            Ok(ret)
        }
    }

    /// # Safety
    /// `params` must be a valid, writable `IoUringParams`.
    pub unsafe fn io_uring_setup(entries: u32, params: *mut IoUringParams) -> SysResult {
        wrap(unsafe { syscall(nr::IO_URING_SETUP, entries, params) })
    }

    /// # Safety
    /// `fd` must be a ring fd returned by [`io_uring_setup`].
    pub unsafe fn io_uring_enter(
        fd: i32,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
    ) -> SysResult {
        wrap(unsafe {
            syscall(
                nr::IO_URING_ENTER,
                fd as i64,
                to_submit as i64,
                min_complete as i64,
                flags as i64,
                0i64,
                0i64,
            )
        })
    }

    /// # Safety
    /// `addr` must be a valid mapping of at least `len` bytes.
    pub unsafe fn madvise(addr: *mut c_void, len: usize, advice: i32) -> SysResult {
        wrap(unsafe { syscall(nr::MADVISE, addr, len, advice as i64) })
    }

    /// # Safety
    /// `out` must be a valid, writable `Cachestat`.
    pub unsafe fn cachestat(
        fd: i32,
        range: *const CachestatRange,
        out: *mut Cachestat,
    ) -> SysResult {
        wrap(unsafe { syscall(nr::CACHESTAT, fd as i64, range, out, 0i64) })
    }

    /// # Safety
    /// `iov` must point to `iovcnt` valid iovecs with writable buffers.
    pub unsafe fn preadv2(
        fd: i32,
        iov: *const IoVec,
        iovcnt: i32,
        offset: i64,
        flags: i32,
    ) -> SysResult {
        // The offset is split into low/high words on 32-bit ABIs; on the 64-bit
        // targets this crate supports it is passed whole, with -1 meaning "use the
        // file position".
        wrap(unsafe {
            syscall(
                nr::PREADV2,
                fd as i64,
                iov,
                iovcnt as i64,
                offset,
                0i64,
                flags as i64,
            )
        })
    }

    /// # Safety
    /// Standard `mmap` contract.
    pub unsafe fn mmap(
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: u64,
    ) -> Result<*mut c_void, i32> {
        let r = unsafe {
            syscall(
                nr::MMAP,
                std::ptr::null_mut::<c_void>(),
                len,
                prot as i64,
                flags as i64,
                fd as i64,
                offset as i64,
            )
        };
        if r == MAP_FAILED as i64 || r < 0 {
            let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(if e != 0 { e } else { (-r) as i32 });
        }
        Ok(r as *mut c_void)
    }

    /// # Safety
    /// `addr`/`len` must describe a mapping created by [`mmap`].
    pub unsafe fn munmap(addr: *mut c_void, len: usize) -> SysResult {
        wrap(unsafe { syscall(nr::MUNMAP, addr, len) })
    }
}

/// Fail-closed stubs for targets without these syscalls.
///
/// Every function has the same signature as its Linux counterpart and returns
/// `ENOSYS` without executing anything. This keeps the crate compiling — and its
/// tests runnable — on macOS and Windows while guaranteeing no foreign syscall is
/// ever issued.
#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;

    /// # Safety
    /// Never dereferences its arguments; present for signature compatibility.
    pub unsafe fn io_uring_setup(_entries: u32, _params: *mut IoUringParams) -> SysResult {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn io_uring_enter(_fd: i32, _to_submit: u32, _min: u32, _flags: u32) -> SysResult {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn madvise(_addr: *mut c_void, _len: usize, _advice: i32) -> SysResult {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn cachestat(
        _fd: i32,
        _range: *const CachestatRange,
        _out: *mut Cachestat,
    ) -> SysResult {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn preadv2(
        _fd: i32,
        _iov: *const IoVec,
        _cnt: i32,
        _off: i64,
        _flags: i32,
    ) -> SysResult {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn mmap(
        _len: usize,
        _prot: i32,
        _flags: i32,
        _fd: i32,
        _off: u64,
    ) -> Result<*mut c_void, i32> {
        Err(ENOSYS)
    }
    /// # Safety
    /// As above.
    pub unsafe fn munmap(_addr: *mut c_void, _len: usize) -> SysResult {
        Err(ENOSYS)
    }
}

pub use imp::*;

/// Whether this build targets a platform whose syscalls are implemented.
pub const SUPPORTED_PLATFORM: bool = cfg!(target_os = "linux");
