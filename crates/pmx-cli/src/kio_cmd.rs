// SPDX-License-Identifier: GPL-2.0-or-later
//! `potatomaxx kio` — measure the kernel I/O paths against each other.
//!
//! The project's central claim is that queue depth dominates: the bandwidth
//! surface shows 6-8x between one outstanding read and eight. Until now that was
//! only ever *modelled*. This measures it three ways on the same device, same
//! corpus, same request pattern:
//!
//! * synchronous `pread` — the depth-1 baseline, and what a demand-paged
//!   `mmap` effectively does;
//! * a thread pool — depth by spending a thread and a syscall per read;
//! * io_uring — depth by submitting a batch with one syscall.
//!
//! and then repeats the io_uring pass with `RWF_DONTCACHE` to show what the page
//! cache was costing.

use pmx_kio::hints;
use pmx_kio::sys;
use pmx_kio::{Capabilities, ReadOp, Ring};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.0} {}", U[i])
    }
}

/// Deterministic pseudo-random offsets, so every engine reads the same blocks.
fn offsets(n: usize, slots: u64, blob: u64, seed: u64) -> Vec<u64> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 16) % slots * blob
        })
        .collect()
}

fn write_corpus(path: &Path, bytes: u64) -> std::io::Result<()> {
    if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) == bytes {
        return Ok(());
    }
    let mut f = File::create(path)?;
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

/// One measured configuration.
struct Row {
    engine: &'static str,
    qd: usize,
    gbps: f64,
    dontcache: bool,
    /// Share of the corpus resident in the page cache when this cell started.
    resident_before: f64,
}

/// Available RAM in bytes, from `/proc/meminfo`.
///
/// Needed because a buffered-read benchmark over a corpus that fits in RAM
/// measures the page cache, not the device — and will happily report bandwidths
/// several times the hardware's ceiling.
fn available_ram() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Share of `path` currently in the page cache.
fn resident_fraction(path: &Path, corpus: u64) -> f64 {
    File::open(path)
        .ok()
        .and_then(|f| hints::residency(&f, 0, 0).ok())
        .map(|r| r.cached_fraction(corpus / 4096))
        .unwrap_or(f64::NAN)
}

/// Synchronous pread, one read at a time.
fn bench_pread(path: &Path, blob: u64, slots: u64, budget: Duration, flags: i32) -> f64 {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };
    let mut buf = vec![0u8; blob as usize];
    let offs = offsets(1 << 16, slots, blob, 7);
    let t0 = Instant::now();
    let mut done = 0u64;
    let mut i = 0usize;
    while t0.elapsed() < budget {
        for _ in 0..8 {
            if hints::pread_flags(&f, &mut buf, offs[i % offs.len()], flags).is_err() {
                return 0.0;
            }
            i += 1;
            done += 1;
        }
    }
    (done * blob) as f64 / t0.elapsed().as_secs_f64()
}

/// A thread pool: depth by spending a thread per outstanding read.
fn bench_threads(
    path: &Path,
    blob: u64,
    slots: u64,
    qd: usize,
    budget: Duration,
    flags: i32,
) -> f64 {
    let f = match File::open(path) {
        Ok(f) => Arc::new(f),
        Err(_) => return 0.0,
    };
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(qd);
    for t in 0..qd {
        let f = Arc::clone(&f);
        handles.push(std::thread::spawn(move || -> u64 {
            let mut buf = vec![0u8; blob as usize];
            let offs = offsets(1 << 14, slots, blob, 7 + t as u64);
            let mut done = 0u64;
            let mut i = 0usize;
            while t0.elapsed() < budget {
                for _ in 0..8 {
                    if hints::pread_flags(&*f, &mut buf, offs[i % offs.len()], flags).is_err() {
                        return done;
                    }
                    i += 1;
                    done += 1;
                }
            }
            done
        }));
    }
    let total: u64 = handles.into_iter().filter_map(|h| h.join().ok()).sum();
    (total * blob) as f64 / t0.elapsed().as_secs_f64()
}

/// io_uring, sliding window: keep `qd` reads in flight continuously.
///
/// The distinction from `bench_uring` is the whole point of measuring both. That
/// one drains the ring between batches; this one replaces each completed read
/// immediately, so the device never sees an empty queue.
fn bench_uring_stream(
    path: &Path,
    blob: u64,
    slots: u64,
    qd: usize,
    budget: Duration,
    flags: i32,
) -> f64 {
    let mut ring = match Ring::new((qd.max(1) as u32 * 2).max(8)) {
        Ok(r) => r,
        Err(_) => return 0.0,
    };
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };
    let mut bufs: Vec<Vec<u8>> = (0..qd.max(1)).map(|_| vec![0u8; blob as usize]).collect();
    // Chunk the offset list so the budget can be checked without abandoning
    // reads that are already in flight.
    let chunk = (qd * 16).max(64);
    let t0 = Instant::now();
    let mut done = 0u64;
    let mut seed = 11u64;
    while t0.elapsed() < budget {
        seed = seed.wrapping_add(1);
        let offs = offsets(chunk, slots, blob, seed);
        match ring.read_many(&f, &offs, blob as u32, qd, &mut bufs, flags) {
            Ok(rs) => {
                if rs.iter().any(|r| r.res < 0) {
                    return 0.0;
                }
                done += rs.len() as u64;
            }
            Err(_) => return 0.0,
        }
    }
    (done * blob) as f64 / t0.elapsed().as_secs_f64()
}

/// io_uring: depth by submitting a batch with one syscall.
fn bench_uring(path: &Path, blob: u64, slots: u64, qd: usize, budget: Duration, flags: i32) -> f64 {
    let mut ring = match Ring::new(qd.max(1) as u32 * 2) {
        Ok(r) => r,
        Err(_) => return 0.0,
    };
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };
    let mut storage: Vec<Vec<u8>> = (0..qd).map(|_| vec![0u8; blob as usize]).collect();
    let offs = offsets(1 << 14, slots, blob, 11);
    let t0 = Instant::now();
    let mut done = 0u64;
    let mut i = 0usize;
    while t0.elapsed() < budget {
        let ops: Vec<ReadOp> = (0..qd)
            .map(|s| ReadOp {
                offset: offs[(i + s) % offs.len()],
                len: blob as u32,
                slot: s as u32,
            })
            .collect();
        i += qd;
        let mut views: Vec<&mut [u8]> = storage.iter_mut().map(|v| v.as_mut_slice()).collect();
        match ring.read_batch(&f, &ops, &mut views, flags) {
            Ok(rs) => {
                if rs.iter().any(|r| r.res < 0) {
                    return 0.0;
                }
                done += rs.len() as u64;
            }
            Err(_) => return 0.0,
        }
    }
    (done * blob) as f64 / t0.elapsed().as_secs_f64()
}

/// Run the comparison.
pub fn run(dir: &str, corpus_mib: u64, blob_kib: u64, ms: u64) -> Result<(), String> {
    let dir = PathBuf::from(dir);
    let path = dir.join("pmx-kio-corpus.bin");
    let corpus = corpus_mib << 20;
    let blob = blob_kib << 10;
    let budget = Duration::from_millis(ms);

    let caps = Capabilities::probe(Path::new("/etc/hostname"));
    println!("kernel facilities: {}", caps.summary());

    // These are *buffered* reads, unlike `potatomaxx probe`, which uses O_DIRECT.
    // That is the point — buffered is what a real engine uses — but it means a
    // corpus that fits in RAM measures the page cache rather than the device.
    if let Some(ram) = available_ram() {
        if corpus < ram * 2 {
            println!(
                "\nWARNING: the {} corpus is smaller than twice this machine's {} of RAM.\n\
                 These are buffered reads, so most of it will sit in the page cache and the\n\
                 numbers below will measure memory, not storage — expect figures above the\n\
                 device's ceiling. Re-run with --corpus-mib {} or more for a device measurement.",
                human(corpus),
                human(ram),
                (ram * 2) >> 20
            );
        }
    }
    if !caps.io_uring {
        println!("\nio_uring is unavailable here, so only the synchronous paths will be measured.");
    }
    println!(
        "\ncorpus {} in {}, {} requests, {} ms per cell",
        human(corpus),
        dir.display(),
        human(blob),
        ms
    );
    write_corpus(&path, corpus).map_err(|e| format!("writing the corpus: {e}"))?;
    let slots = (corpus / blob).max(1);

    // Probe RWF_DONTCACHE against the corpus itself: support is per-filesystem.
    let dontcache_ok = File::open(&path)
        .map(|f| hints::dontcache_supported(&f))
        .unwrap_or(false);

    // Build the cell list, then run it in an interleaved order. Running every
    // io_uring cell after every pread cell would let cache warm-up show up as an
    // io_uring win, which is exactly the mistake this ordering avoids.
    #[allow(clippy::type_complexity)]
    let mut cells: Vec<(&'static str, usize, bool)> = vec![("pread", 1, false)];
    for qd in [4usize, 8, 16] {
        cells.push(("threads", qd, false));
    }
    if caps.io_uring {
        for qd in [4usize, 8, 16, 32] {
            cells.push(("uring-batch", qd, false));
        }
        for qd in [4usize, 8, 16, 32, 64] {
            cells.push(("uring-stream", qd, false));
        }
        if dontcache_ok {
            for qd in [16usize, 32] {
                cells.push(("uring-stream", qd, true));
            }
        }
    }
    // Deterministic interleave: alternate from the front and back of the list.
    let mut order: Vec<usize> = Vec::with_capacity(cells.len());
    let (mut lo, mut hi) = (0usize, cells.len());
    while lo < hi {
        order.push(lo);
        lo += 1;
        if lo < hi {
            hi -= 1;
            order.push(hi);
        }
    }

    let mut rows: Vec<Row> = Vec::with_capacity(cells.len());
    for &ci in &order {
        let (engine, qd, dc) = cells[ci];
        let flags = if dc { sys::RWF_DONTCACHE } else { 0 };
        let before = if caps.cachestat {
            resident_fraction(&path, corpus)
        } else {
            f64::NAN
        };
        let bps = match engine {
            "pread" => bench_pread(&path, blob, slots, budget, flags),
            "threads" => bench_threads(&path, blob, slots, qd, budget, flags),
            "uring-batch" => bench_uring(&path, blob, slots, qd, budget, flags),
            _ => bench_uring_stream(&path, blob, slots, qd, budget, flags),
        };
        rows.push(Row {
            engine,
            qd,
            gbps: bps / 1e9,
            dontcache: dc,
            resident_before: before,
        });
    }
    // Report in a stable order regardless of execution order.
    rows.sort_by(|a, b| {
        a.engine
            .cmp(b.engine)
            .then(a.dontcache.cmp(&b.dontcache))
            .then(a.qd.cmp(&b.qd))
    });

    println!(
        "\n{:>10} {:>5} {:>11} {:>11} {:>10}",
        "engine", "QD", "GB/s", "cached@start", "flags"
    );
    for r in &rows {
        println!(
            "{:>10} {:>5} {:>11.3} {:>10.0}% {:>10}",
            r.engine,
            r.qd,
            r.gbps,
            r.resident_before * 100.0,
            if r.dontcache { "DONTCACHE" } else { "-" }
        );
    }

    let base = rows
        .iter()
        .find(|r| r.engine == "pread")
        .map(|r| r.gbps)
        .unwrap_or(0.0);
    let best = rows.iter().max_by(|a, b| a.gbps.total_cmp(&b.gbps));
    if let Some(b) = best {
        println!(
            "\nbest: {} at QD{}{} — {:.3} GB/s, {:.1}x the depth-1 baseline",
            b.engine,
            b.qd,
            if b.dontcache { " +DONTCACHE" } else { "" },
            b.gbps,
            b.gbps / base.max(1e-9)
        );
    }
    // Compare the two ways of reaching depth, at matched queue depth.
    let at = |eng: &str, qd: usize, dc: bool| {
        rows.iter()
            .find(|r| r.engine == eng && r.qd == qd && r.dontcache == dc)
            .map(|r| r.gbps)
    };
    // The comparison that matters: two ways of reaching the same nominal depth.
    for qd in [8usize, 16, 32] {
        if let (Some(b), Some(st)) = (at("uring-batch", qd, false), at("uring-stream", qd, false)) {
            println!(
                "QD{qd}: batch-and-drain {b:.3} GB/s vs sliding window {st:.3} GB/s ({:+.0}%) \
                 — same depth, but draining the ring between batches idles the device",
                (st / b.max(1e-9) - 1.0) * 100.0
            );
        }
    }
    for qd in [8usize, 16] {
        if let (Some(t), Some(u)) = (at("threads", qd, false), at("uring-stream", qd, false)) {
            println!(
                "QD{qd}: thread pool {t:.3} GB/s vs io_uring {u:.3} GB/s ({:+.0}%)",
                (u / t.max(1e-9) - 1.0) * 100.0
            );
        }
    }
    if let (Some(plain), Some(dc)) = (at("uring-stream", 16, false), at("uring-stream", 16, true)) {
        println!(
            "\nRWF_DONTCACHE at QD16: {dc:.3} vs {plain:.3} GB/s ({:+.0}%). It is not a \
             throughput\noptimisation — it declines to keep pages nothing will read again, so \
             the cost\nfalls on this process rather than on everything else running.",
            (dc / plain.max(1e-9) - 1.0) * 100.0
        );
    }

    // Recommend from the measurement rather than from a preference. io_uring is
    // the better interface in principle — fewer syscalls, no thread per
    // outstanding read — but a single ring driven from one thread loses to a
    // thread pool on some platforms, virtualised block devices especially. The
    // point of measuring is to find out which case this machine is.
    let best_thread = rows
        .iter()
        .filter(|r| r.engine == "threads" && !r.dontcache)
        .max_by(|a, b| a.gbps.total_cmp(&b.gbps));
    let best_uring = rows
        .iter()
        .filter(|r| r.engine == "uring-stream" && !r.dontcache)
        .max_by(|a, b| a.gbps.total_cmp(&b.gbps));
    if let (Some(t), Some(u)) = (best_thread, best_uring) {
        println!();
        if t.gbps > u.gbps * 1.05 {
            println!(
                "RECOMMENDATION for this machine: a thread pool at QD{} ({:.3} GB/s), not \
                 io_uring\n({:.3} GB/s at QD{}). A single ring submits and reaps from one \
                 thread; where the\nblock path is virtualised or per-request cost dominates, \
                 more submitting threads win.\nio_uring would need SQPOLL or a ring per thread \
                 to close that gap.",
                t.qd, t.gbps, u.gbps, u.qd
            );
        } else if u.gbps > t.gbps * 1.05 {
            println!(
                "RECOMMENDATION for this machine: io_uring at QD{} ({:.3} GB/s) over a thread \
                 pool\n({:.3} GB/s at QD{}) — one syscall per batch and no thread per \
                 outstanding read.",
                u.qd, u.gbps, t.gbps, t.qd
            );
        } else {
            println!(
                "RECOMMENDATION: the two engines are within 5% here ({:.3} vs {:.3} GB/s), so \
                 prefer\nthe thread pool for its simpler failure modes.",
                t.gbps, u.gbps
            );
        }
        println!(
            "\nWhat is not in doubt is queue depth: {:.3} GB/s at depth 1 against {:.3} at the \
             best\nconfiguration, a factor of {:.0}. That is the lever worth engineering for, \
             and it is\nwhy expert prefetch prediction matters — you cannot queue reads you \
             have not predicted.",
            base,
            t.gbps.max(u.gbps),
            t.gbps.max(u.gbps) / base.max(1e-9)
        );
    }

    // What the page cache is holding afterwards is the other half of the story:
    // a streaming read that leaves nothing behind is a read that did not evict
    // somebody else's working set.
    if caps.cachestat {
        if let Ok(f) = File::open(&path) {
            if let Ok(res) = hints::residency(&f, 0, 0) {
                let pages = corpus / 4096;
                println!(
                    "\npage cache after the run: {} of {} pages resident ({:.1}%){}",
                    res.cached,
                    pages,
                    res.cached_fraction(pages) * 100.0,
                    if res.is_thrashing() {
                        ", and thrashing — pages evicted then wanted back"
                    } else {
                        ""
                    }
                );
                println!(
                    "That residency is the cost a streaming read imposes on everything else \
                     on the machine.\nRWF_DONTCACHE is how you decline to pay it."
                );
            }
        }
    }
    let _ = std::fs::remove_file(&path);
    Ok(())
}
