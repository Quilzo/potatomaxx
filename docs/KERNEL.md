# potatomaxx and the Linux kernel

This document exists because "put it in the kernel" is a reasonable-sounding goal
that, taken literally, would waste months and end in a rejected patch series. It
sets out what cannot go in, what can, and what a realistic contribution looks
like.

## The short version

**An inference engine will not be merged into Linux, and should not be.** But
there is real, upstreamable work adjacent to it, and the kernel has spent the last
few years building precisely the extension points this workload needs.

## Why inference cannot live in the kernel

### Floating point is prohibited

Kernel code may not use FP registers or the `float`/`double` types. This is not
stylistic: the kernel does not save userspace FP state on entry, so touching FP
registers silently corrupts whatever userspace process was interrupted. The
[floating-point API](https://docs.kernel.org/core-api/floating-point.html)
provides `kernel_fpu_begin()`/`kernel_fpu_end()` for the few places that need it
— some crypto, some RAID — and the documentation is explicit that preemption may
be disabled inside, so **the critical section should be minimised**.

A GEMM is the exact opposite of minimal. Wrapping one in `kernel_fpu_begin()`
would create non-preemptible sections lasting milliseconds, which destroys the
latency guarantees every other subsystem depends on. There is no version of this
that a maintainer accepts, and they would be right.

Integer-only inference does not rescue the idea either. It removes the FP
objection and leaves the larger one.

### A model is policy

"Mechanism in the kernel, policy in userspace" is the project's oldest structural
rule. A quantised model, a routing predictor, and a cache-admission heuristic are
all policy — they encode judgements about a workload that the kernel has no
business holding an opinion about.

The instructive precedent is IBM's **ML-LIB** RFC, which proposed a machine
learning library *for* the kernel. Even that far more modest proposal keeps the
models in userspace and puts only "ML model proxy" components in kernel space, and
it was flagged on arrival as likely to be contentious. If a proxy layer is
contentious, an inference engine is not a conversation.

### What the kernel says to do instead

The kernel's answer to "we need application-specific policy in a hot path" is
now BPF, and the pattern is well established:

| framework | status | what it makes programmable |
|---|---|---|
| [`sched_ext`](https://www.phoronix.com/news/Linux-6.12-Lands-sched-ext) | **merged in 6.12** | CPU scheduling policy, as BPF |
| [`cache_ext`](https://www.asafcidon.com/uploads/5/9/7/0/59701649/cache_ext.pdf) | research, explicitly modelled on `sched_ext` | page cache eviction policy |
| [FetchBPF](https://www.usenix.org/system/files/atc24-cao.pdf) | research (USENIX ATC '24) | prefetch policy, via new eBPF hooks |
| [DAMON / DAMOS](https://docs.kernel.org/mm/damon/) | **merged**, in production at AWS and SK hynix | access-aware operations, memory tiering |

That is the shape of a legitimate answer: the *policy* runs in the kernel, as a
verified BPF program, loaded by a userspace program that owns the model.

## What the kernel already gives us, and what it is worth

These are used today, probed at runtime rather than inferred from a version
string — distributions backport, containers lie, and `RWF_DONTCACHE` additionally
needs the filesystem to have opted in via `FOP_DONTCACHE`.

| mechanism | since | why this workload wants it |
|---|---|---|
| io_uring batched reads | 5.1 | queue depth with one syscall per batch, not per read |
| `RWF_DONTCACHE` | 6.14 | stream cold weights without evicting everything else |
| `MADV_HUGEPAGE` | long-standing | the resident hot set is read every token; 4 KiB pages are needless TLB pressure |
| `MADV_RANDOM` | long-standing | readahead bets the next block is wanted; a router decides otherwise |
| `cachestat(2)` | 6.5 | verify residency rather than assume it |

### Measured on the development machine

18 GiB corpus against 7.6 GB of RAM, so the page cache holds only ~7% of it and
these are device numbers rather than memory numbers. 64 KiB requests.

| engine | QD1 | QD4 | QD8 | QD16 | QD32 |
|---|---|---|---|---|---|
| `pread` | 0.099 | | | | |
| thread pool | | 0.537 | 1.522 | **3.291** | |
| io_uring, batch-and-drain | | 0.305 | 0.550 | 0.816 | 1.086 |
| io_uring, sliding window | | 1.101 | 1.362 | 1.544 | 1.063 |
| io_uring + `RWF_DONTCACHE` | | | | 0.795 | 0.816 |

Three things worth taking from this.

**Queue depth is worth 33x** — 0.099 GB/s to 3.291. That is the lever, and it
dwarfs everything else this project can do. It is also why expert prefetch
prediction matters: you cannot queue reads you have not predicted.

**How you drive io_uring matters more than using it.** Submitting a batch and
waiting for all of it drains the ring between batches, so the device idles in the
gap; that configuration peaked at QD8 and then declined. Switching to a sliding
window — replace each completed read immediately — gained **+148% at QD8** and
+89% at QD16.

**And io_uring still lost to a thread pool here**, 1.54 against 3.29 GB/s at
QD16. One ring submitting and reaping from a single thread is not enough where
the block path is virtualised (this is WSL2) or where per-request cost dominates;
sixteen threads simply have more submitters. Closing that would need `SQPOLL`, a
ring per thread, or `IORING_SETUP_IOPOLL` on a native device. Reporting it the
other way round would have been easy and wrong, so `potatomaxx kio` recommends
from the measurement rather than from a preference for an interface.

`RWF_DONTCACHE` deserves particular note. Jens Axboe's series reported **65–75%
higher throughput at half the CPU** on exactly this access pattern — a large
sequential-ish stream read once. Without it, reading a 40 GB expert store fills
the page cache with data that will not be reused, evicts every other process's
working set, and then makes reclaim the bottleneck. On this machine it measured **49% slower**, which is the honest result and not an
argument against it: it is not a throughput optimisation. It declines to retain
pages nothing will read again — 7.4% page-cache residency instead of filling RAM
— so the cost of streaming a 40 GB store falls on this process rather than on
every other process on the machine. `potatomaxx kio` reports both the bandwidth
and the residency, because reporting only the first would make the flag look
strictly bad.

## What a real contribution looks like

Three tiers, honestly ranked by how likely they are to land.

### Tier 1 — BPF policies (runs in the kernel, needs no kernel patch)

This is genuinely "kernel level", and it is the sanctioned route.

- **A `sched_ext` scheduler for inference on hybrid CPUs.** On a P-core/E-core
  part, the compute threads want P-cores and the prefetch threads emphatically do
  not — they are I/O-bound and belong on E-cores, out of the way. The default
  fair scheduler cannot know that. `sched_ext` exists so it does not have to.
- **A `cache_ext`-style eviction policy** carrying the expert-frequency
  information `pmx-plan` already computes. The measurements in this repo show LRU
  reaching only 80.9% of the offline optimum on skewed expert routing where a
  frequency-and-cost-aware policy reaches 90.7% — and the page cache cannot know
  an expert's routing frequency, because that is application knowledge.
- **A FetchBPF-style prefetch policy** driven by router lookahead.

The honest catch: `sched_ext` is merged but the two page-cache frameworks are
research prototypes, so those need their patches or an upstreaming effort.

### Tier 2 — a workload that justifies upstreaming

`cache_ext` and FetchBPF both argue that applications know things the page cache
cannot. Expert routing is an unusually clean example: the access distribution is
strongly skewed, the application knows the distribution exactly, and the kernel
cannot infer it. A reproducible benchmark demonstrating that, with numbers, is a
real contribution to those efforts — and cheaper than writing a subsystem.

### Tier 3 — an upstream mechanism

If something belongs in the kernel proper, it is a *general* mechanism with more
than one user. Candidates worth exploring, in rough order of plausibility:

- **A batched, deadline-aware prefetch hint.** `MADV_WILLNEED` and
  `posix_fadvise(POSIX_FADV_WILLNEED)` are one range at a time with no way to say
  "these forty ranges, and I need them within a millisecond". Sparse-model
  inference is not the only workload with that shape — graph traversal and vector
  search share it.
- **A DAMON access source fed by application hints**, so DAMOS tiering can act on
  what the application knows rather than only on what sampling observes.

Neither is close to a patch, and either would need a second independent user
before it deserved one. Saying so is more useful than pretending otherwise.

## What this repo will not do

- Add FP to kernel code, under any wrapper.
- Ship an out-of-tree kernel module that duplicates a userspace job. A module
  that could be a userspace program is a maintenance burden with a security
  surface and no upside.
- Claim a kernel patch is coming when what exists is a BPF program and a
  benchmark.

## Testing status

`pmx-kio` is exercised on Linux 6.18. The `sched_ext` and `cache_ext` work above
is **not** implemented here, and would be dishonest to ship untested: this
development machine reports `CONFIG_SCHED_CLASS_EXT is not set`, and has neither
`bpftool` nor `pahole`, so a BPF scheduler could be written but not run. That is a
prerequisite to be satisfied, not a detail to gloss over.

| facility | development machine (6.18, WSL2) |
|---|---|
| io_uring | available, exercised |
| `RWF_DONTCACHE` | available, exercised |
| `cachestat(2)` | available, exercised |
| THP (`madvise` mode) | available, exercised |
| `sched_ext` | **absent** — `CONFIG_SCHED_CLASS_EXT` not set |
| DAMON | **absent** |
| bpftool, pahole | **absent** |
