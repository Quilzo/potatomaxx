# potatomaxx

**Make a mixture-of-experts model cheaper to *read*, so it runs on hardware that
cannot hold it.**

A big MoE checkpoint does not fit in RAM, so it is read from storage while it
runs. That makes decoding a storage problem, and storage cares enormously about
*how* you ask. On the development machine — an i5-1235U laptop, 7.6 GB RAM, no
GPU — the same NVMe delivers **0.099 GB/s at queue depth 1 and 3.29 GB/s at depth
16**. A factor of **33**, from nothing but concurrency.

`potatomaxx` attacks that in three ways, and measures each rather than asserting
it:

1. **Layout** — reorder experts on disk so the ones that fire together are
   adjacent, turning scattered slice reads into contiguous runs. Output is a
   **drop-in GGUF**: same tensor names, same shapes, byte-identical weights.
   llama.cpp, Colibri and anything else read it unchanged.
2. **Precision** — store each expert at its own bit width, chosen from *measured*
   quantisation error and how often the expert is actually used. 3.58× less
   weight movement at 0.31× the size on the synthetic model.
3. **Prefetch** — predict which experts a layer will select before its router has
   run, because you cannot queue reads you have not predicted.

Zero dependencies, `std` only. 160 tests. GPL-2.0-or-later.

```console
$ potatomaxx kio
     engine    QD        GB/s cached@start      flags
      pread     1       0.099          3%          -
    threads    16       3.291          4%          -
uring-stream    16       1.544          5%          -
uring-stream    16       0.795          4%  DONTCACHE

best: threads at QD16 — 3.291 GB/s, 33.1x the depth-1 baseline
```

---

## Install

```bash
make                       # cargo build --release
sudo make install          # honours prefix and DESTDIR
make check                 # tests, debug and release
```

GNU Makefile conventions are followed (`prefix`, `DESTDIR`, `install`,
`uninstall`, `check`, `dist`, `info`), so distribution packaging should be
uneventful. `man potatomaxx` and `info potatomaxx` are both provided.

## Quick start, no model download required

```bash
cargo build --release

potatomaxx synth                        # synthetic MoE checkpoint + routing trace
potatomaxx kio                          # which kernel I/O path is fastest here
potatomaxx probe --out surf.json        # this device's bandwidth surface
potatomaxx analyse --model synth.gguf --trace synth.pmxtrace --probe surf.json
potatomaxx plan    --model synth.gguf --trace synth.pmxtrace --probe surf.json --out p.pmxplan
potatomaxx pack    --model synth.gguf --plan p.pmxplan --out packed.gguf
potatomaxx verify  --model synth.gguf --repacked packed.gguf --plan p.pmxplan
```

Twelve commands, all exercised in CI: `probe`, `kio`, `predict`, `build-store`,
`bench`, `synth`, `inspect`, `analyse`, `plan`, `pack`, `verify`, `help`.

## Why a repack is safe

Permuting the expert axis is a **relabelling**. If new slot `j` holds old expert
`perm[j]`, and the router's weight rows are permuted by the same `perm`, then the
logit computed for slot `j` is exactly the logit the original model computed for
expert `perm[j]`. Top-k selects the same real experts, from the same bytes. The
function computed is unchanged, bit for bit.

The GGUF specification permits the file-level half of this: tensor data is located
only through the `offset` in its `tensor_info`, physical order is unconstrained,
and inter-tensor padding is explicitly allowed.

`potatomaxx verify` proves it afterwards, reading both files in full and comparing
every expert slice. It is strict — flipping four bytes anywhere fails it, and CI
asserts that.

```console
$ potatomaxx verify --model synth.gguf --repacked packed.gguf --plan p.pmxplan
  13 tensors compared, 1 byte-identical, 12 matched as a permutation
  12.10 MiB of weights confirmed unchanged
OK — the repacked file holds exactly the original weights.
```

## What is measured, and what is not

This distinction is the point of the project, so it comes before the features.

| Claim | Status |
|---|---|
| Bandwidth surface, engine comparison, queue-depth scaling | **Measured** on the development machine |
| `verify` proves byte-identical weights | **Measured** — tests, plus a tamper control in CI |
| Predictor recall | **Measured** on the given trace, held out from fitting |
| Cache policy comparison | **Measured** against the offline optimum |
| Per-expert precision: size and movement | **Measured** (error), **computed** (movement, from the surface) |
| Expert-fetch throughput (`bench`) | **Computed** from the measured surface + a replayed trace |
| Speedup on *your* model | **Unknown until you run `analyse`** — that is what it is for |
| End-to-end generated tokens/sec | **Not measured.** There is no attention, KV cache or sampler here |

`bench` reports a *ceiling* on decode rate, not a decode rate. On a memory-bound
machine that ceiling is the binding constraint, which is why it was built first.
Quality is measured as round-trip error, which is **not** perplexity: a precision
plan that looks cheap by RMSE still needs a real eval before production use.

## Findings that contradicted the design

Kept because they are the most useful output, and each is now a regression test.

**Layout is the secondary lever.** The original design argued for co-activation
disk layout as the primary win. Measurement disagreed: once a layer's top-k reads
are issued concurrently, reordering buys 1.1–1.6×, and only for slices in a
particular size band. Queue depth is worth 6–33×, and no file layout can
influence it.

**Fixed groups read whole are a net loss.** Over-reading a 2 MiB group costs more
than coalescing saves above ~256 KiB slices. Only coalescing the slices you
actually need pays.

**GDSF beats LRU — but only when cost per byte varies.** It reaches 90.7% of the
offline optimum against LRU's 80.9% on skewed routing. Its key contains
`cost / size`, though, so at a fixed bandwidth (`cost = bytes / rate`) that term
is constant and it collapses to LFU — which is *worse* than LRU. Cost-awareness
pays across storage tiers, not within one.

**Prefetching is not free.** Every prediction is a real read, charged whether the
router wants it or not. Throughput peaks near `top_k` and then falls while recall
keeps climbing. In one configuration plain on-demand LRU beat every prefetch
setting, so `bench --compare` sweeps rather than declaring a winner.

**A loss budget relative to the baseline is degenerate.** Once sensitivity is
measured the baseline *is* the reference and its error is zero — so any multiple
of it permits nothing, and the allocator refuses every demotion while still
reporting success. Replaced with a uniform-precision ceiling (`--error-bits`).

**io_uring lost to a thread pool here.** Driven as batch-and-drain it peaked at
QD8 and then declined; fixing that to a sliding window gained +148% at QD8, and it
still lost to 16 threads (1.54 vs 3.29 GB/s). One ring submitting and reaping from
one thread is not enough where the block path is virtualised. `kio` therefore
*recommends from the measurement* instead of preferring an interface.

**`RWF_DONTCACHE` is 49% slower, and worth it anyway.** It is not a throughput
optimisation. It declines to retain pages nothing will read again — 7.4% page-cache
residency instead of filling RAM — so the cost of a streaming read falls on this
process rather than on everything else on the machine.

## Features

### Layout compiler

`analyse` scores the checkpoint's existing expert order against an optimised one,
costed against a measured bandwidth surface, and says plainly when the gain is not
worth rewriting a file for. `plan` writes a reviewable text plan; `pack` applies
it; `verify` proves it.

### Per-expert precision

Impossible in GGUF — a tensor carries one `ggml_type` and a MoE layer's experts
share a tensor — so `build-store` emits a native `.pmxstore`. Bits follow
**access frequency and tier cost**, from real per-expert round-trip error:

```
block  0: 128 experts, 9.0% resident, movement 3.58x faster, expected error 0.02001
wrote 3072 slices: 14.26 MiB of weights plus 2.84 MiB alignment padding
weights are 0.31x the source for the same experts
precision mix: pmxq3=60 pmxq4=588 pmxq8=120
```

Hot experts keep 8 bits, cold ones drop to 3. The nearest prior art,
[APEX-Quant](https://github.com/localai-org/apex-quant), allocates by structural
role and layer position and uses no runtime traces.

**Routers are never requantised.** Quantisation error in a router perturbs expert
*selection* — the "expert shift" problem — which would invalidate the very trace
the plan came from.

### Router lookahead

Training-free, using only routing history a running engine already has:

| predictor | recall @ 1× top-k | @ 2× | @ 4× |
|---|---|---|---|
| frequency (baseline) | 0.100 | 0.199 | 0.386 |
| sticky | 0.415 | 0.483 | 0.604 |
| **sticky+markov** | **0.415** | **0.629** | **0.739** |

Chance is 0.100. 0.739 at 4× budget is roughly what PILOT-style single-layer
lookahead reports (71.6%), with no trained head. Trained pre-attention routers
reach 93–98% and it would be dishonest to imply parity. `predict` scores whichever
you use on your own trace.

### Kernel I/O

Probed at runtime, never inferred from a version string — distributions backport,
containers lie, and `RWF_DONTCACHE` additionally needs the filesystem to have
opted in via `FOP_DONTCACHE`.

| mechanism | since | why this workload wants it |
|---|---|---|
| io_uring, sliding window | 5.1 | depth without a thread per outstanding read |
| `RWF_DONTCACHE` | 6.14 | stream cold weights without evicting everything else |
| `MADV_HUGEPAGE` | — | the resident hot set is read every token |
| `MADV_RANDOM` | — | readahead bets the next block is wanted; a router decides otherwise |
| `cachestat(2)` | 6.5 | verify residency rather than assume it |

See [docs/KERNEL.md](docs/KERNEL.md) for what can and cannot be upstreamed, and
why an inference engine does not belong in the kernel.

## Getting a real trace

`potatomaxx` needs to know which experts *your* workload selects. The text format
is one line per `(token, layer)`, so patching an engine to emit it is a small
change:

```
# token layer experts...
0 0 12 44 7 91
0 1 3 55 8 12
1 0 12 44 9 91
```

Traces from your workload beat a generic calibration corpus: routing is
workload-specific, and so is the layout that suits it.

## Prior art

The runtime side of disk-resident MoE inference is well covered, and this is not
another runtime:

- **[Colibri](https://github.com/JustVugg/colibri)** — pure C, expert streaming,
  router-lookahead prefetch, learned pinning, speculative decoding. The mature
  option; if you want a runtime, start there.
- **[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)** —
  sparsity-aware expert cache.
- **llama.cpp [#25294](https://github.com/ggml-org/llama.cpp/pull/25294)** —
  SSD-backed expert streaming with `O_DIRECT` and a slot cache.
- **Oracle-MoE**, **Sticky Routing**, **ReMoE** — improve locality by changing
  *routing*, at training time.

`potatomaxx` changes the **byte layout and precision of the file**, so it composes
with all of the above rather than competing, and risks nothing: the weights are
provably unchanged.

## Security

Model files are untrusted input from public hubs, and the parser is the part of an
inference stack with **no performance requirement at all**. This format's recent
history is a run of memory-safety failures in exactly that code path:

- **CVE-2026-27940** — integer overflow in llama.cpp's
  `gguf_init_from_file_impl()` producing an undersized heap allocation, then a
  528+ byte controlled overflow. A bypass of the fix for CVE-2025-53630.
- **CVE-2026-7482** ("Bleeding Llama", CVSS 9.1) — out-of-bounds read from
  inflated tensor dimensions in Ollama's loader, leaking process memory.

In safe Rust the first is a checked-arithmetic error and the second a bounds
check — but only if the code actually uses checked arithmetic and validates
against the real file size. So there is a test suite that throws hostile input at
the parser: 1,400 random inputs, every single-byte mutation of a valid header,
truncation at every length, inflated counts and dimensions, unaligned and
out-of-range offsets, invalid UTF-8. The contract is absolute — **any input at
all yields `Ok` or `Err`, never a panic and never a read outside the file.**

Every crate is `#![forbid(unsafe_code)]` except three, each with a stated reason
and its invariants written down:

| crate | why `unsafe` | how it is checked |
|---|---|---|
| `pmx-probe` | page-aligned buffers for `O_DIRECT` | alignment test |
| `pmx-kernels` | SIMD intrinsics | scalar path is authoritative; every vector path tested against it |
| `pmx-kio` | raw syscalls, shared io_uring mappings | batched reads compared byte-for-byte against `std` reads; the non-Linux path is built and tested too |

## Layout

| crate | responsibility | `unsafe` |
|---|---|---|
| `pmx-gguf` | GGUF read, offset rewriting, permutation, verification | forbidden |
| `pmx-kernels` | GGUF dequantisation, native block formats, SIMD int8 dot | audited |
| `pmx-kio` | io_uring, `RWF_DONTCACHE`, huge pages, residency | audited |
| `pmx-probe` | device bandwidth surface (blob size × queue depth) | audited |
| `pmx-trace` | trace format, co-activation statistics, synthetic traces | forbidden |
| `pmx-partition` | expert-order optimisation against the measured surface | forbidden |
| `pmx-plan` | residency and bit allocation by frequency × tier cost | forbidden |
| `pmx-store` | native store: per-expert precision, contiguous experts | forbidden |
| `pmx-cache` | expert residency cache: LRU, LFU, GDSF | forbidden |
| `pmx-predict` | router lookahead, training-free | forbidden |
| `pmx-runtime` | replay harness tying prefetch, cache and precision together | forbidden |
| `pmx-cli` | the `potatomaxx` binary | forbidden |

## Testing

```bash
cargo test                  # 160 tests, debug: integer overflow checks on
cargo test --release        # 160 tests, release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

No fixtures, no network, no GPU. `potatomaxx synth` builds everything the suite
needs in seconds.

Bugs the suite caught during development, as a guide to what is and is not
settled: a factor-of-two error in half-precision subnormal decode (found by
sweeping all 65,536 bit patterns); a store index whose writer and reader
disagreed by four bytes per record; non-deterministic cache eviction from
`HashMap` iteration order; 25 MiB of alignment padding around 6 MiB of weights;
a `NaN` sensitivity from one non-finite weight that silently disabled precision
allocation while reporting success; a benchmark that measured the page cache
instead of the device because its corpus fit in RAM; syscall numbers gated on
architecture but not OS, so a macOS build issued a Linux syscall and died with
`SIGSYS`; and a capability probe that reported `ENOSYS` as "supported" because it
treated every unexpected errno as success.

**Platform support.** `pmx-kio` needs Linux — io_uring, `RWF_DONTCACHE` and
`cachestat(2)` have no portable equivalent. On other targets every syscall
wrapper fails closed with `ENOSYS` and emits no syscall instruction, so the crate
still builds and its suite still runs; CI covers Linux and macOS, and the
non-Linux path is exercised deliberately rather than assumed.

## Status and how to help

Early, and honest about it. Everything above runs, but it has been exercised on
synthetic checkpoints and one laptop. The most valuable contributions right now:

- **Run it against a real MoE checkpoint** and report the `analyse` and
  `bench --compare` output — especially where it says the gain is not worth it.
- **Run `potatomaxx kio` on real hardware.** Every conclusion here follows from
  the shape of one laptop's bandwidth surface. SATA SSDs, eMMC, spinning disks,
  Apple unified memory and bare-metal NVMe will each say something different, and
  the io_uring-versus-threads result in particular is likely platform-specific.
- **A quality evaluation** for the precision allocator, so RMSE can be replaced
  by perplexity.
- **The kernel-side BPF work** in [docs/KERNEL.md](docs/KERNEL.md): a `sched_ext`
  scheduler, a `cache_ext`-style eviction policy. Written but untested here —
  this machine reports `CONFIG_SCHED_CLASS_EXT is not set`.

Patches want a `Signed-off-by:` line (`git commit -s`), per the kernel's
Developer's Certificate of Origin. See [CONTRIBUTING.md](CONTRIBUTING.md),
[GOVERNANCE.md](GOVERNANCE.md) and [SECURITY.md](SECURITY.md).

On foundations: [docs/SUBMISSION.md](docs/SUBMISSION.md) records which ones can
actually accept a project like this and why most cannot yet — the ASF requires
Apache-2.0, the SFC requires a project over a year old with an established
community, and LF AI & Data requires paid membership. GNU is the one that
evaluates work in progress, and the materials are prepared there.

## Licence

**GPL-2.0-or-later.** Chosen for compatibility rather than preference: the Linux
kernel's [licensing rules](https://docs.kernel.org/process/license-rules.html)
list `GPL-2.0+` among the compatible licences, and permit dual licensing with
MIT, BSD and Apache-2.0. AGPL appears nowhere on that list, and is incompatible
with combining GPL-2.0-only or Apache-2.0 code — which would rule out both
kernel-tree inclusion and composing with llama.cpp (MIT) or Colibri
(Apache-2.0). Every source file carries an SPDX identifier.
