# Changelog

Notable changes, newest first. This project follows
[Semantic Versioning](https://semver.org/); before 1.0 the API and the on-disk
`.pmxstore` format may change between minor versions.

Entries marked **finding** record a measurement that contradicted the design.
They are kept because they are the most useful thing here.

## Unreleased

### Real-model validation

First run against a real mixture-of-experts checkpoint (IBM Granite 3.0
1b-a400m, 822 MB, 24 layers x 32 experts, top-8). Full results in
[docs/REAL-MODEL-RESULTS.md](docs/REAL-MODEL-RESULTS.md).

- Repack verified byte-identical across 96 permuted real Q4_K and Q6_K tensors;
  782 MiB of weights confirmed unchanged, file size identical. A four-byte
  corruption was rejected and localised to the exact tensor and slice.
- New `compare` subcommand validates dequantisers differentially. Against the
  F16 build of the same model as ground truth: Q4_K correlation 0.997089
  (7.63% relative error), Q6_K 0.999822 (1.89%) — the ratio their bit-widths
  predict, from decoders that share no code.
- **finding** On this model `analyse` reported no useful gain on every layer,
  correctly: 420 KiB expert slices sit above the point where request size stops
  mattering. The store also came out 0.99x the source, because granite is
  already Q4_K. Recorded because a tool that only reports wins cannot be
  trusted when it reports one.
- **finding** 2 MiB group alignment cost 106.75 MiB of padding around 690 MiB
  of weights (~15%) on this model — a straightforward loss here, which is why
  padding is reported separately from payload rather than folded in.

## 0.1.0 — 2026-08-22

First public version. Everything runs end to end; nothing has yet been run
against a real MoE checkpoint.

### Layout compiler

- `analyse`, `plan`, `pack`, `verify`: reorder experts on disk by co-activation,
  emitting a **drop-in GGUF** — same tensor names and shapes, byte-identical
  weights. Permuting the expert axis is a relabelling, so permuting the router's
  rows identically leaves the computed function unchanged bit for bit.
- `verify` proves this afterwards by comparing both files in full, and is strict
  enough to reject a four-byte flip. CI asserts that.
- **finding** Layout is the *secondary* lever. Once a layer's top-k reads are
  issued concurrently, reordering buys 1.1–1.6×, and only for slices in a
  particular size band.
- **finding** Fixed groups read whole are a net loss above ~256 KiB slices: the
  over-read exceeds what coalescing saves.

### Per-expert precision

- `build-store` emits a native `.pmxstore` with each expert at its own bit width,
  allocated from **measured** per-expert round-trip error and access frequency.
  3.58× less weight movement at 0.31× the size on the synthetic model.
- Routers are never requantised — quantisation error there perturbs expert
  selection, invalidating the trace the plan came from.
- **finding** A loss budget expressed as a multiple of the baseline is degenerate
  once sensitivity is measured: the baseline is the reference, its error is zero,
  and any multiple of zero permits nothing while still reporting success.
  Replaced with `--error-bits`, a uniform-precision ceiling.

### Router lookahead

- `predict` scores four training-free predictors. `sticky+markov` reaches 0.739
  recall at 4× budget against 0.100 chance — roughly what PILOT-style lookahead
  reports, with no trained head.
- **finding** Prefetching is not free. Every prediction is a real read, so
  throughput peaks near `top_k` and then falls while recall keeps climbing. In one
  configuration plain on-demand LRU beat every prefetch setting.

### Expert cache

- LRU, LFU and GDSF, measured against the offline optimum. GDSF reaches 90.7% of
  it against LRU's 80.9% on skewed routing.
- **finding** GDSF's advantage vanishes when fetch cost is proportional to size:
  its `cost / size` term becomes constant and it collapses to LFU, which is
  *worse* than LRU. Cost-awareness pays across storage tiers, not within one.

### Kernel I/O

- `kio` compares `pread`, a thread pool and io_uring, with `RWF_DONTCACHE`,
  `MADV_HUGEPAGE`, `MADV_RANDOM` and `cachestat(2)`. io_uring is implemented by
  hand, so the zero-dependency property survives.
- **Queue depth is worth 33×** — 0.099 GB/s at depth 1 against 3.291 at depth 16,
  on an 18 GiB corpus against 7.6 GB of RAM.
- **finding** How io_uring is driven matters more than using it: batch-and-drain
  peaked at QD8 then declined; a sliding window gained +148% at QD8.
- **finding** io_uring still lost to a thread pool here (1.54 vs 3.29 GB/s at
  QD16). One ring on one thread is not enough on a virtualised block path, so
  `kio` recommends from the measurement rather than from a preference.
- **finding** `RWF_DONTCACHE` measured 49% slower and is worth it anyway: 7.4%
  page-cache residency instead of filling RAM. It is not a throughput
  optimisation.

### Security and correctness

- Hostile-input suite for the parser: 1,400 random inputs, every single-byte
  header mutation, truncation at every length, inflated counts and dimensions,
  unaligned and out-of-range offsets, invalid UTF-8. Contract: any input yields
  `Ok` or `Err`, never a panic and never an out-of-bounds read.
- 162 tests, run in debug as well as release so integer overflow checks apply.
- Every crate `#![forbid(unsafe_code)]` except `pmx-kernels`, `pmx-kio` and
  `pmx-probe`, each with written invariants.

### Bugs found and fixed during development

- Factor-of-two error in half-precision subnormal decode, found by sweeping all
  65,536 bit patterns.
- Store index whose writer emitted 36 bytes per record while the reader strode 40.
- Non-deterministic cache eviction from `HashMap` iteration order.
- 25 MiB of alignment padding around 6 MiB of weights.
- A `NaN` sensitivity, from one non-finite weight, that silently disabled
  precision allocation while reporting success.
- A benchmark that measured the page cache instead of the device because its
  corpus fit in RAM.
- Syscall numbers gated on architecture but not OS, so a macOS build issued a
  Linux syscall and died with `SIGSYS`.
- A capability probe that reported `ENOSYS` as "supported" because it treated any
  unexpected errno as success.

### Licence

- GPL-2.0-or-later, with an SPDX identifier on every file. Chosen for
  compatibility: the kernel's licensing rules list `GPL-2.0+` as compatible and
  permit dual licensing with MIT, BSD and Apache-2.0, while AGPL appears nowhere
  on that list and cannot combine with GPL-2.0-only or Apache-2.0 code.
