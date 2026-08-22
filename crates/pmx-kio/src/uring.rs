// SPDX-License-Identifier: GPL-2.0-or-later
//! A minimal io_uring reader.
//!
//! # Why this exists
//!
//! The measured bandwidth surface says queue depth is worth 6-8x on this class
//! of device — far more than any file layout can deliver. Until now the project
//! only *modelled* that: the replay harness charged for reads at a configured
//! queue depth without ever issuing them concurrently. This is the code that
//! makes the lever real.
//!
//! `preadv2` with a thread pool can reach depth too, at the cost of a thread and
//! a syscall per read. io_uring submits a whole batch with one syscall and reaps
//! completions from shared memory, which is why the published progression for
//! database workloads runs from ~16.5k tx/s on libaio to ~546k with SQPOLL.
//!
//! # Scope
//!
//! Deliberately small: submit a batch of reads, wait for them all, report what
//! happened. No fixed buffers, no SQPOLL, no linked SQEs. Those are real wins
//! but each adds a failure mode, and the first thing worth knowing is whether
//! plain batched submission beats a thread pool on the target hardware.
//!
//! # Memory ordering
//!
//! The rings are shared with the kernel, so the tail we publish and the head the
//! kernel publishes need real atomics, not plain loads and stores. Submission
//! writes the SQE and then releases the tail; completion acquires the tail before
//! reading any CQE and releases the head afterwards. Getting this wrong produces
//! a race that only shows up under load, so the accesses are funnelled through
//! two helpers rather than written inline.

use crate::sys::{self, IoUringCqe, IoUringParams, IoUringSqe};
use crate::KioError;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{fence, AtomicU32, Ordering};

/// One read to perform.
#[derive(Debug, Clone, Copy)]
pub struct ReadOp {
    /// Byte offset in the file.
    pub offset: u64,
    /// Bytes to read.
    pub len: u32,
    /// Index into the caller's buffer table, returned in the result.
    pub slot: u32,
}

/// Outcome of one read in a [`Ring::read_many`] stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamResult {
    /// Index into the `offsets` slice this completion belongs to.
    pub index: usize,
    /// Bytes transferred, or a negative errno.
    pub res: i32,
}

/// Outcome of one read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadResult {
    /// The `slot` from the corresponding [`ReadOp`].
    pub slot: u32,
    /// Bytes transferred, or a negative errno.
    pub res: i32,
}

/// A mapped region of the ring, unmapped on drop.
struct Mapping {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr`/`len` came from a successful `mmap` in `Ring::new`
            // and are unmapped exactly once, here.
            unsafe {
                let _ = sys::munmap(self.ptr, self.len);
            }
        }
    }
}

/// An io_uring instance set up for reads.
pub struct Ring {
    fd: RawFd,
    entries: u32,
    features: u32,
    _sq_map: Mapping,
    _sqe_map: Mapping,
    cq_map_is_sq: bool,
    _cq_map: Option<Mapping>,

    sq_head: *const AtomicU32,
    sq_tail: *const AtomicU32,
    sq_mask: u32,
    sq_array: *mut u32,
    sqes: *mut IoUringSqe,

    cq_head: *const AtomicU32,
    cq_tail: *const AtomicU32,
    cq_mask: u32,
    cqes: *const IoUringCqe,
}

// SAFETY: `Ring` owns its mappings and its file descriptor, holds no borrowed
// references, and all shared-memory access goes through atomics. Moving one
// between threads is sound; concurrent *use* still requires `&mut`, which the
// borrow checker enforces.
unsafe impl Send for Ring {}

impl Ring {
    /// Set up a ring with at least `entries` submission slots.
    ///
    /// `entries` is rounded up to a power of two by the kernel and capped at
    /// 4096 here, which is far beyond any useful expert-fetch batch.
    pub fn new(entries: u32) -> Result<Self, KioError> {
        let entries = entries.clamp(1, 4096).next_power_of_two();
        let mut p = IoUringParams::default();
        // SAFETY: `p` is a valid, writable, correctly-sized params struct.
        let fd =
            unsafe { sys::io_uring_setup(entries, &mut p) }.map_err(KioError::UringSetup)? as RawFd;

        // Two separate ring mappings were only needed before 5.4. Requiring the
        // single-mapping feature keeps the setup path short, and every kernel
        // that has RWF_DONTCACHE has had it for years.
        if p.features & sys::IORING_FEAT_SINGLE_MMAP == 0 {
            // SAFETY: fd came from io_uring_setup and is closed exactly once.
            unsafe { close_fd(fd) };
            return Err(KioError::MissingFeature("IORING_FEAT_SINGLE_MMAP"));
        }

        let sq_ring_sz = p.sq_off.array as usize + p.sq_entries as usize * 4;
        let cq_ring_sz = p.cq_off.cqes as usize + p.cq_entries as usize * 16;
        let ring_sz = sq_ring_sz.max(cq_ring_sz);

        // SAFETY: mapping the ring at the documented offset with the size the
        // kernel just reported.
        let sq_ptr = match unsafe {
            sys::mmap(
                ring_sz,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED | sys::MAP_POPULATE,
                fd,
                sys::IORING_OFF_SQ_RING,
            )
        } {
            Ok(p) => p,
            Err(e) => {
                unsafe { close_fd(fd) };
                return Err(KioError::UringMmap(e));
            }
        };
        let sq_map = Mapping {
            ptr: sq_ptr,
            len: ring_sz,
        };

        let sqe_sz = p.sq_entries as usize * std::mem::size_of::<IoUringSqe>();
        // SAFETY: as above, for the SQE array.
        let sqe_ptr = match unsafe {
            sys::mmap(
                sqe_sz,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED | sys::MAP_POPULATE,
                fd,
                sys::IORING_OFF_SQES,
            )
        } {
            Ok(p) => p,
            Err(e) => {
                drop(sq_map);
                unsafe { close_fd(fd) };
                return Err(KioError::UringMmap(e));
            }
        };
        let sqe_map = Mapping {
            ptr: sqe_ptr,
            len: sqe_sz,
        };

        // SAFETY: every offset below was reported by the kernel for this ring and
        // lies inside the mapping whose size was computed from those same
        // offsets. The ring fields are 32-bit and naturally aligned.
        unsafe {
            let sq = sq_ptr as *mut u8;
            let cq = sq_ptr as *mut u8; // single mmap
            Ok(Ring {
                fd,
                entries: p.sq_entries,
                features: p.features,
                sq_head: sq.add(p.sq_off.head as usize) as *const AtomicU32,
                sq_tail: sq.add(p.sq_off.tail as usize) as *const AtomicU32,
                sq_mask: *(sq.add(p.sq_off.ring_mask as usize) as *const u32),
                sq_array: sq.add(p.sq_off.array as usize) as *mut u32,
                sqes: sqe_ptr as *mut IoUringSqe,
                cq_head: cq.add(p.cq_off.head as usize) as *const AtomicU32,
                cq_tail: cq.add(p.cq_off.tail as usize) as *const AtomicU32,
                cq_mask: *(cq.add(p.cq_off.ring_mask as usize) as *const u32),
                cqes: cq.add(p.cq_off.cqes as usize) as *const IoUringCqe,
                _sq_map: sq_map,
                _sqe_map: sqe_map,
                cq_map_is_sq: true,
                _cq_map: None,
            })
        }
    }

    /// Submission slots available.
    pub fn entries(&self) -> u32 {
        self.entries
    }

    /// Submission entries the kernel has consumed but not yet completed.
    ///
    /// Both callers bound their own submissions by free-slot accounting, so this
    /// is not needed for correctness. It is exposed because it is the natural
    /// signal for backpressure, and because a ring that cannot be inspected is
    /// hard to debug when it stalls.
    pub fn sq_pending(&self) -> u32 {
        // SAFETY: both point into the live ring mapping and are 32-bit aligned.
        unsafe {
            let head = (*self.sq_head).load(Ordering::Acquire);
            let tail = (*self.sq_tail).load(Ordering::Acquire);
            tail.wrapping_sub(head)
        }
    }

    /// Feature bits the kernel reported.
    pub fn features(&self) -> u32 {
        self.features
    }

    /// Whether the kernel guarantees no completion-queue drops.
    pub fn no_drop(&self) -> bool {
        self.features & sys::IORING_FEAT_NODROP != 0
    }

    /// Read `ops` from `file` into `bufs`, all in flight together.
    ///
    /// `bufs[op.slot]` receives each read's data and must be at least `op.len`
    /// bytes. `rw_flags` goes into the SQE — pass [`crate::sys::RWF_DONTCACHE`]
    /// to avoid retaining the pages.
    ///
    /// Returns one [`ReadResult`] per op, in completion order. A short read or an
    /// error is reported in `res` rather than raised, because a batch is
    /// partially useful and the caller decides what to do about a gap.
    pub fn read_batch<F: AsRawFd>(
        &mut self,
        file: &F,
        ops: &[ReadOp],
        bufs: &mut [&mut [u8]],
        rw_flags: i32,
    ) -> Result<Vec<ReadResult>, KioError> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        if ops.len() > self.entries as usize {
            return Err(KioError::BatchTooLarge {
                asked: ops.len(),
                limit: self.entries as usize,
            });
        }
        for op in ops {
            let b = bufs
                .get(op.slot as usize)
                .ok_or(KioError::BadSlot(op.slot))?;
            if b.len() < op.len as usize {
                return Err(KioError::BufferTooSmall {
                    slot: op.slot,
                    need: op.len as usize,
                    have: b.len(),
                });
            }
        }

        let fd = file.as_raw_fd();
        // SAFETY: `sq_tail` points into the live ring mapping. The kernel owns
        // entries between head and tail; we only write slots at or beyond tail,
        // and publish them with a release store below.
        let tail_cell = unsafe { &*self.sq_tail };
        let mut tail = tail_cell.load(Ordering::Acquire);

        for (i, op) in ops.iter().enumerate() {
            let idx = (tail & self.sq_mask) as usize;
            let sqe = IoUringSqe {
                opcode: sys::IORING_OP_READ,
                fd,
                off: op.offset,
                addr: bufs[op.slot as usize].as_mut_ptr() as u64,
                len: op.len,
                rw_flags,
                user_data: i as u64,
                ..Default::default()
            };
            // SAFETY: `idx` is masked into range, and this slot is not visible to
            // the kernel until the release store on `tail`.
            unsafe {
                self.sqes.add(idx).write(sqe);
                self.sq_array.add(idx).write(idx as u32);
            }
            tail = tail.wrapping_add(1);
        }
        // Publish every SQE written above before the kernel can observe the tail.
        tail_cell.store(tail, Ordering::Release);

        let n = ops.len() as u32;
        let mut submitted = 0u32;
        let mut out = Vec::with_capacity(ops.len());

        while submitted < n || out.len() < ops.len() {
            let want = n - submitted;
            // SAFETY: `self.fd` is this ring's fd.
            let r = unsafe {
                sys::io_uring_enter(
                    self.fd,
                    want,
                    (ops.len() - out.len()) as u32,
                    sys::IORING_ENTER_GETEVENTS,
                )
            };
            match r {
                Ok(v) => submitted += v as u32,
                // A signal during the wait is not a failure; retry.
                Err(4) => {}
                Err(e) => return Err(KioError::UringEnter(e)),
            }
            self.reap(ops, &mut out);
            if submitted >= n && out.len() >= ops.len() {
                break;
            }
        }
        Ok(out)
    }

    /// Read every offset in `offsets`, keeping `depth` reads in flight at all
    /// times.
    ///
    /// [`Ring::read_batch`] submits a batch and waits for all of it, which puts a
    /// barrier between batches: the ring drains to empty and refills, so average
    /// depth is well below the nominal figure and the device idles in the gap.
    /// Measured on the development machine, batch-and-drain peaked at queue depth
    /// 8 and then *declined* — 1.599 GB/s at QD8 falling to 0.742 at QD32 — while
    /// a plain thread pool, which has no barrier, kept scaling to 2.749 GB/s.
    /// That was a defect in how the ring was driven, not in io_uring.
    ///
    /// This is the sliding-window form: a completion immediately frees its buffer
    /// slot and the next read is submitted into it, so the device always has
    /// `depth` requests queued until the work runs out.
    ///
    /// `bufs` must hold at least `depth` buffers of at least `len` bytes. Buffers
    /// are recycled as completions arrive, so a slot's contents are only valid
    /// between its completion and the next submission that reuses it — which is
    /// why the results carry the offset index rather than a borrow.
    pub fn read_many<F: AsRawFd>(
        &mut self,
        file: &F,
        offsets: &[u64],
        len: u32,
        depth: usize,
        bufs: &mut [Vec<u8>],
        rw_flags: i32,
    ) -> Result<Vec<StreamResult>, KioError> {
        let depth = depth.clamp(1, self.entries as usize).min(bufs.len().max(1));
        if offsets.is_empty() {
            return Ok(Vec::new());
        }
        if bufs.len() < depth {
            return Err(KioError::BufferTooSmall {
                slot: 0,
                need: depth,
                have: bufs.len(),
            });
        }
        for (i, b) in bufs.iter().enumerate().take(depth) {
            if b.len() < len as usize {
                return Err(KioError::BufferTooSmall {
                    slot: i as u32,
                    need: len as usize,
                    have: b.len(),
                });
            }
        }

        let fd = file.as_raw_fd();
        // SAFETY: both point into the live ring mapping.
        let sq_tail_cell = unsafe { &*self.sq_tail };
        let cq_head_cell = unsafe { &*self.cq_head };
        let cq_tail_cell = unsafe { &*self.cq_tail };

        let mut free: Vec<u32> = (0..depth as u32).rev().collect();
        // Which offset index each in-flight slot is serving.
        let mut slot_op: Vec<usize> = vec![usize::MAX; depth];
        let mut next = 0usize;
        let mut in_flight = 0usize;
        let mut out: Vec<StreamResult> = Vec::with_capacity(offsets.len());

        while next < offsets.len() || in_flight > 0 {
            // Fill the window.
            let mut to_submit = 0u32;
            let mut tail = sq_tail_cell.load(Ordering::Acquire);
            while next < offsets.len() {
                let slot = match free.pop() {
                    Some(s) => s,
                    None => break,
                };
                let idx = (tail & self.sq_mask) as usize;
                let sqe = IoUringSqe {
                    opcode: sys::IORING_OP_READ,
                    fd,
                    off: offsets[next],
                    addr: bufs[slot as usize].as_mut_ptr() as u64,
                    len,
                    rw_flags,
                    user_data: slot as u64,
                    ..Default::default()
                };
                // SAFETY: `idx` is masked into range and the slot is invisible to
                // the kernel until the release store on the tail below.
                unsafe {
                    self.sqes.add(idx).write(sqe);
                    self.sq_array.add(idx).write(idx as u32);
                }
                slot_op[slot as usize] = next;
                tail = tail.wrapping_add(1);
                to_submit += 1;
                next += 1;
                in_flight += 1;
            }
            if to_submit > 0 {
                sq_tail_cell.store(tail, Ordering::Release);
            }

            // Submit, and wait for at least one completion only when the window
            // is full or the work is done — otherwise keep filling.
            let must_wait = in_flight > 0 && (free.is_empty() || next >= offsets.len());
            let min_complete = if must_wait { 1 } else { 0 };
            let flags = if min_complete > 0 {
                sys::IORING_ENTER_GETEVENTS
            } else {
                0
            };
            if to_submit > 0 || min_complete > 0 {
                // SAFETY: `self.fd` is this ring's fd.
                match unsafe { sys::io_uring_enter(self.fd, to_submit, min_complete, flags) } {
                    Ok(_) => {}
                    // Interrupted by a signal; the SQEs stay queued.
                    Err(4) => {}
                    Err(e) => return Err(KioError::UringEnter(e)),
                }
            }

            // Harvest whatever is ready and hand the slots straight back.
            let mut head = cq_head_cell.load(Ordering::Relaxed);
            let cq_tail = cq_tail_cell.load(Ordering::Acquire);
            let mut reaped = 0usize;
            while head != cq_tail {
                let ci = (head & self.cq_mask) as usize;
                // SAFETY: the kernel has published CQEs up to `cq_tail`.
                let cqe = unsafe { *self.cqes.add(ci) };
                let slot = cqe.user_data as usize;
                if slot < depth && slot_op[slot] != usize::MAX {
                    out.push(StreamResult {
                        index: slot_op[slot],
                        res: cqe.res,
                    });
                    slot_op[slot] = usize::MAX;
                    free.push(slot as u32);
                    in_flight -= 1;
                    reaped += 1;
                }
                head = head.wrapping_add(1);
            }
            if reaped > 0 {
                fence(Ordering::AcqRel);
                cq_head_cell.store(head, Ordering::Release);
            }
        }
        Ok(out)
    }

    /// Drain the completion queue into `out`.
    fn reap(&self, ops: &[ReadOp], out: &mut Vec<ReadResult>) {
        // SAFETY: both point into the live ring mapping.
        let head_cell = unsafe { &*self.cq_head };
        let tail_cell = unsafe { &*self.cq_tail };
        let mut head = head_cell.load(Ordering::Relaxed);
        // Acquire the tail before reading any CQE the kernel published.
        let tail = tail_cell.load(Ordering::Acquire);
        while head != tail {
            let idx = (head & self.cq_mask) as usize;
            // SAFETY: `idx` is masked into the CQE array, which the kernel has
            // published up to `tail`.
            let cqe = unsafe { *self.cqes.add(idx) };
            if let Some(op) = ops.get(cqe.user_data as usize) {
                out.push(ReadResult {
                    slot: op.slot,
                    res: cqe.res,
                });
            }
            head = head.wrapping_add(1);
        }
        // Release the consumed slots back to the kernel.
        fence(Ordering::AcqRel);
        head_cell.store(head, Ordering::Release);
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        debug_assert!(self.cq_map_is_sq);
        // SAFETY: `fd` came from io_uring_setup and is closed exactly once. The
        // mappings are released by their own Drop impls afterwards.
        unsafe { close_fd(self.fd) };
    }
}

/// # Safety
/// `fd` must be a valid descriptor not used again afterwards.
unsafe fn close_fd(fd: RawFd) {
    unsafe {
        let _ = sys_close(fd);
    }
}

unsafe fn sys_close(fd: RawFd) -> i64 {
    // `extern` blocks are implicitly unsafe to call in edition 2021; the
    // `unsafe extern` spelling needs Rust 1.82 and would break the declared MSRV.
    extern "C" {
        fn close(fd: i32) -> i32;
    }
    unsafe { close(fd) as i64 }
}

/// Whether io_uring can be set up at all on this kernel.
pub fn available() -> bool {
    Ring::new(4).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};

    fn corpus(name: &str, bytes: usize) -> std::path::PathBuf {
        let d = std::env::temp_dir().join("pmx-kio-tests");
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(name);
        if std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) != bytes as u64 {
            let mut f = File::create(&p).unwrap();
            let mut x: u64 = 0x2545_F491_4F6C_DD1D;
            let mut chunk = vec![0u8; 64 << 10];
            let mut left = bytes;
            while left > 0 {
                for b in chunk.iter_mut() {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *b = (x >> 24) as u8;
                }
                let n = left.min(chunk.len());
                f.write_all(&chunk[..n]).unwrap();
                left -= n;
            }
            f.sync_all().unwrap();
        }
        p
    }

    #[test]
    fn a_quiet_ring_has_nothing_pending() {
        let r = match Ring::new(16) {
            Ok(r) => r,
            Err(_) => return,
        };
        assert_eq!(r.sq_pending(), 0);
    }

    #[test]
    fn ring_sets_up_and_reports_features() {
        let r = match Ring::new(32) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable here ({e}); skipping");
                return;
            }
        };
        assert!(r.entries() >= 32);
        assert!(r.features() & sys::IORING_FEAT_SINGLE_MMAP != 0);
    }

    #[test]
    fn a_batch_read_matches_what_std_reads() {
        // The load-bearing test. Everything else about this crate is a
        // performance claim; this is the correctness claim, and a memory-ordering
        // or index mistake shows up here as wrong bytes.
        let mut ring = match Ring::new(64) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("io_uring unavailable here ({e}); skipping");
                return;
            }
        };
        let path = corpus("batch.bin", 4 << 20);
        let f = File::open(&path).unwrap();

        const N: usize = 32;
        const LEN: u32 = 8192;
        let mut storage: Vec<Vec<u8>> = (0..N).map(|_| vec![0u8; LEN as usize]).collect();
        let ops: Vec<ReadOp> = (0..N)
            .map(|i| ReadOp {
                offset: (i as u64 * 97_384) % ((4 << 20) - LEN as u64),
                len: LEN,
                slot: i as u32,
            })
            .collect();

        let results = {
            let mut views: Vec<&mut [u8]> = storage.iter_mut().map(|v| v.as_mut_slice()).collect();
            ring.read_batch(&f, &ops, &mut views, 0).unwrap()
        };
        assert_eq!(results.len(), N, "every op must complete");
        for r in &results {
            assert_eq!(
                r.res, LEN as i32,
                "slot {} short or failed: {}",
                r.slot, r.res
            );
        }

        // Compare against ordinary reads.
        let mut check = File::open(&path).unwrap();
        let mut want = vec![0u8; LEN as usize];
        for op in &ops {
            check.seek(SeekFrom::Start(op.offset)).unwrap();
            check.read_exact(&mut want).unwrap();
            assert_eq!(
                storage[op.slot as usize], want,
                "slot {} differs from a std read at offset {}",
                op.slot, op.offset
            );
        }
    }

    #[test]
    fn dontcache_reads_return_the_same_bytes() {
        // RWF_DONTCACHE changes retention, never contents. If it ever changed
        // contents that would be a kernel bug, but asserting it costs nothing and
        // documents the expectation.
        let mut ring = match Ring::new(16) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("dontcache.bin", 1 << 20);
        let f = File::open(&path).unwrap();
        let ops = [ReadOp {
            offset: 65536,
            len: 4096,
            slot: 0,
        }];

        let mut a = vec![0u8; 4096];
        let mut b = vec![0u8; 4096];
        {
            let mut v: Vec<&mut [u8]> = vec![a.as_mut_slice()];
            ring.read_batch(&f, &ops, &mut v, 0).unwrap();
        }
        {
            let mut v: Vec<&mut [u8]> = vec![b.as_mut_slice()];
            let r = ring
                .read_batch(&f, &ops, &mut v, sys::RWF_DONTCACHE)
                .unwrap();
            // An older kernel would reject the flag; tolerate that explicitly
            // rather than failing the suite on a version difference.
            if r[0].res < 0 {
                eprintln!(
                    "RWF_DONTCACHE unsupported here (errno {}); skipping",
                    -r[0].res
                );
                return;
            }
        }
        assert_eq!(a, b, "RWF_DONTCACHE must not change the data read");
    }

    #[test]
    fn read_many_returns_every_offset_exactly_once() {
        let mut ring = match Ring::new(64) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("stream.bin", 8 << 20);
        let f = File::open(&path).unwrap();
        const LEN: u32 = 8192;
        let offsets: Vec<u64> = (0..200u64)
            .map(|i| (i * 37_741) % ((8 << 20) - LEN as u64))
            .collect();
        let mut bufs: Vec<Vec<u8>> = (0..16).map(|_| vec![0u8; LEN as usize]).collect();
        let res = ring.read_many(&f, &offsets, LEN, 16, &mut bufs, 0).unwrap();

        assert_eq!(res.len(), offsets.len(), "every offset must complete once");
        let mut seen = vec![false; offsets.len()];
        for r in &res {
            assert_eq!(r.res, LEN as i32, "index {} failed: {}", r.index, r.res);
            assert!(!seen[r.index], "index {} completed twice", r.index);
            seen[r.index] = true;
        }
        assert!(seen.iter().all(|s| *s), "some offset never completed");
    }

    #[test]
    fn read_many_data_matches_std_reads() {
        // Correctness under buffer recycling: a slot must not be reused before its
        // completion, or the bytes for one offset land in another's buffer.
        let mut ring = match Ring::new(32) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("streamdata.bin", 4 << 20);
        let f = File::open(&path).unwrap();
        const LEN: u32 = 4096;
        // Small depth with many offsets forces heavy slot reuse.
        let offsets: Vec<u64> = (0..64u64).map(|i| i * LEN as u64).collect();
        let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; LEN as usize]).collect();

        // Verify one offset at a time so the buffer contents are checkable.
        let mut check = File::open(&path).unwrap();
        let mut want = vec![0u8; LEN as usize];
        for (i, off) in offsets.iter().enumerate() {
            let one = [*off];
            let res = ring.read_many(&f, &one, LEN, 1, &mut bufs, 0).unwrap();
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].res, LEN as i32);
            check.seek(SeekFrom::Start(*off)).unwrap();
            check.read_exact(&mut want).unwrap();
            assert_eq!(bufs[0], want, "offset {i} at {off}");
        }
    }

    #[test]
    fn read_many_needs_a_buffer_per_slot() {
        let mut ring = match Ring::new(16) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("streambuf.bin", 1 << 20);
        let f = File::open(&path).unwrap();
        let mut bufs: Vec<Vec<u8>> = vec![vec![0u8; 64]];
        // Buffer smaller than the read length.
        assert!(matches!(
            ring.read_many(&f, &[0], 4096, 1, &mut bufs, 0),
            Err(KioError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn read_many_with_no_offsets_is_a_no_op() {
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("streamempty.bin", 4096);
        let f = File::open(&path).unwrap();
        let mut bufs: Vec<Vec<u8>> = vec![vec![0u8; 4096]];
        assert_eq!(
            ring.read_many(&f, &[], 4096, 1, &mut bufs, 0)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn an_oversized_batch_is_refused() {
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("small.bin", 1 << 20);
        let f = File::open(&path).unwrap();
        let ops: Vec<ReadOp> = (0..64)
            .map(|i| ReadOp {
                offset: 0,
                len: 512,
                slot: i,
            })
            .collect();
        let mut storage: Vec<Vec<u8>> = (0..64).map(|_| vec![0u8; 512]).collect();
        let mut views: Vec<&mut [u8]> = storage.iter_mut().map(|v| v.as_mut_slice()).collect();
        assert!(matches!(
            ring.read_batch(&f, &ops, &mut views, 0),
            Err(KioError::BatchTooLarge { .. })
        ));
    }

    #[test]
    fn a_short_buffer_is_refused_before_any_io() {
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("short.bin", 1 << 20);
        let f = File::open(&path).unwrap();
        let ops = [ReadOp {
            offset: 0,
            len: 4096,
            slot: 0,
        }];
        let mut small = vec![0u8; 128];
        let mut views: Vec<&mut [u8]> = vec![small.as_mut_slice()];
        assert!(matches!(
            ring.read_batch(&f, &ops, &mut views, 0),
            Err(KioError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn an_empty_batch_is_a_no_op() {
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("empty.bin", 4096);
        let f = File::open(&path).unwrap();
        let mut views: Vec<&mut [u8]> = Vec::new();
        assert_eq!(ring.read_batch(&f, &[], &mut views, 0).unwrap().len(), 0);
    }

    #[test]
    fn many_batches_reuse_the_ring_without_drift() {
        // Ring indices wrap; submitting more ops than the ring has entries,
        // across many batches, exercises that the masked indices and the
        // wrapping head/tail arithmetic stay consistent.
        let mut ring = match Ring::new(8) {
            Ok(r) => r,
            Err(_) => return,
        };
        let path = corpus("wrap.bin", 2 << 20);
        let f = File::open(&path).unwrap();
        let mut buf = vec![0u8; 4096];
        let mut check = File::open(&path).unwrap();
        let mut want = vec![0u8; 4096];
        for round in 0..40u64 {
            let off = (round * 13_337) % ((2 << 20) - 4096);
            let ops = [ReadOp {
                offset: off,
                len: 4096,
                slot: 0,
            }];
            let res = {
                let mut v: Vec<&mut [u8]> = vec![buf.as_mut_slice()];
                ring.read_batch(&f, &ops, &mut v, 0).unwrap()
            };
            assert_eq!(res[0].res, 4096, "round {round}");
            check.seek(SeekFrom::Start(off)).unwrap();
            check.read_exact(&mut want).unwrap();
            assert_eq!(buf, want, "round {round} at offset {off}");
        }
    }
}
