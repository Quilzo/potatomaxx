# potatomaxx

**A layout compiler and streaming toolkit for disk-resident mixture-of-experts models.**

`potatomaxx` reads a GGUF MoE checkpoint and a routing trace, works out an expert
order that turns scattered slice reads into contiguous runs, and rewrites the
file. The output is a **drop-in GGUF** — same tensor names, same shapes,
byte-identical weights, different offsets — so whatever engine you already use
reads it unchanged and gets faster.

Beyond layout it carries the streaming machinery that layout alone cannot reach:
per-expert mixed precision, router-lookahead prefetch, a cost-aware expert cache,
and a replay harness that measures the fetch path end to end.

It also tells you, honestly, when there is nothing to gain — and it did exactly
that several times while being built. The measurements below include the ones
that contradicted the design.

```console
$ potatomaxx analyse --model qwen3-30b-a3b.gguf --trace mine.pmxtrace --probe pmx-probe.json

 layer   experts    req/token        after   speedup  verdict
     0       128        16.24         7.64     2.00x  repack
     1       128        16.24         7.63     1.99x  repack
     2       128        16.19         7.56     2.00x  repack

mean predicted speedup in expert read time: 2.00x
```

---

## Why this exists

A GGUF MoE layer stacks its experts into a few tensors — usually
`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` — with the expert index as the
last axis. Fetching the top-k experts for one token therefore means reading `k`
slices out of each: `k × n_tensors` scattered requests, at whatever size one
expert's slice happens to be.

Storage does not serve all request shapes equally. Measured on the development
machine (i5-1235U laptop, NVMe, `O_DIRECT`, random offsets):

|    blob |   QD1 |   QD4 |   QD8 |  QD16 |
|--------:|------:|------:|------:|------:|
|   4 KiB |  0.02 |  0.06 |  0.09 |  0.13 |
|  16 KiB |  0.06 |  0.16 |  0.25 |  0.48 |
|  32 KiB |  0.08 |  0.35 |  0.57 |  0.83 |
|  64 KiB |  0.15 |  0.38 |  0.61 |  0.91 |
| 256 KiB |  0.35 |  1.24 |  1.82 |  2.16 |
|   1 MiB |  0.91 |  2.46 |  2.51 |  2.61 |
|   2 MiB |  1.03 |  2.09 |  2.54 |  2.67 |
|   8 MiB |  1.55 |  2.40 |  2.26 |  2.18 |
|  32 MiB |  1.95 |  2.38 |  1.92 |  1.17 |

*(GB/s. Reproduce with `potatomaxx probe`.)*

> **Platform note.** Cache bypass uses `O_DIRECT`, which is Linux-only — macOS
> would need `fcntl(F_NOCACHE)` and therefore a `libc` dependency this workspace
> does not take. Elsewhere the probe still runs but measures *cached* reads,
> which can be several times the device's real speed. A surface records which
> happened, and both `probe` and `analyse` warn when the numbers are cached, so
> they can't be silently mistaken for device figures.

Two things fall out of that table:

- **Request size matters enormously below 256 KiB.** At queue depth 8, going
  from 16 KiB to 256 KiB requests is worth **7.3×**. Fine-grained MoE
  checkpoints — hundreds of experts per layer — land squarely in that bad
  region once quantised.
- **Queue depth is worth 6–8× on its own,** and no file layout can influence it.
- **Some devices offer nothing.** Throttled cloud storage measures nearly flat
  across request sizes — CI runners here report ~0.41 GB/s from 64 KiB to 32 MiB.
  On a flat surface no layout can help, and `analyse` says so. That is the tool
  working correctly, and the reason the threshold is a flag (`--min-speedup`)
  rather than a constant.

So layout is the *secondary* lever. `potatomaxx` does the part a file can do:
coalesce the reads. The primary lever — keeping many reads in flight — belongs
to an inference runtime, and this project deliberately does not try to be one.

## Why permuting experts is safe

Permuting the expert axis is a **relabelling**. If new slot `j` holds old expert
`perm[j]`, and the router's weight rows are permuted by the same `perm`, then the
logit computed for slot `j` is exactly the logit the original model computed for
expert `perm[j]`. Top-k selects the same real experts, and each selected slot
holds that expert's original bytes. The function computed is unchanged, bit for
bit.

The GGUF spec permits the file-level half of this: tensor data is located only
through the `offset` in its `tensor_info`, physical order is unconstrained, and
padding between tensors is explicitly allowed.

`potatomaxx verify` proves it after the fact, reading both files in full and
checking every tensor slice-by-slice. It is strict — flipping four bytes anywhere
in a repacked file fails it.

```console
$ potatomaxx verify --model synth.gguf --repacked synth.pmx.gguf --plan synth.pmxplan
  13 tensors compared, 1 byte-identical, 12 matched as a permutation
  9.08 MiB of weights confirmed unchanged
OK — the repacked file holds exactly the original weights.
```

## Try it without downloading a model

```bash
cargo build --release
cd /tmp

potatomaxx synth                      # small synthetic MoE checkpoint + trace
potatomaxx probe --out surf.json      # measure your device (writes then deletes a scratch file)
potatomaxx inspect synth.gguf
potatomaxx analyse --model synth.gguf --trace synth.pmxtrace --probe surf.json
potatomaxx plan    --model synth.gguf --trace synth.pmxtrace --probe surf.json --out p.pmxplan
potatomaxx pack    --model synth.gguf --plan p.pmxplan --out synth.pmx.gguf
potatomaxx verify  --model synth.gguf --repacked synth.pmx.gguf --plan p.pmxplan
```

The plan is a text file. It authorises rewriting a multi-gigabyte model, so you
should be able to read exactly what will happen before it does:

```
pmxplan 1
model qwen3-30b-a3b.gguf
trace mine.pmxtrace

layer 0 experts 128 speedup 1.9991 requests 16.24 -> 7.64
tensors blk.0.ffn_gate_exps.weight,blk.0.ffn_up_exps.weight,blk.0.ffn_down_exps.weight,blk.0.ffn_gate_inp.weight
perm 34,55,53,10,56,43,42,20,46,62,8,0,12,4,61,3,...
```

## Getting a real trace

`potatomaxx` needs to know which experts your workload actually selects. The
text format is one line per `(token, layer)`, so patching an engine to emit it is
a small change:

```
# token layer experts...
0 0 12 44 7 91
0 1 3 55 8 12
1 0 12 44 9 91
```

Feed it in with `--trace`. Traces from *your* workload beat a generic
calibration corpus: routing is workload-specific, and so is the layout that
suits it.

## The streaming path

Layout keeps a checkpoint a drop-in GGUF. Everything else needs a container GGUF
cannot express — **per-expert precision is impossible in GGUF**, because a tensor
carries one `ggml_type` and a MoE layer's experts share a tensor. So there are two
paths, deliberately separate:

| path | output | consumed by | gives you |
|---|---|---|---|
| layout only | drop-in GGUF | llama.cpp, Colibri, anything | fewer, larger reads |
| + per-expert precision | `.pmxstore` | `potatomaxx bench` | fewer *bytes* |

```bash
potatomaxx predict     --trace mine.pmxtrace              # which lookahead works
potatomaxx build-store --model m.gguf --trace mine.pmxtrace --out m.pmxstore
potatomaxx bench       --store m.pmxstore --trace mine.pmxtrace --compare
```

### Per-expert precision, allocated from measured error

`build-store` dequantises each expert, measures the round-trip error of *every*
candidate precision on that expert's real weights, and allocates bits to minimise
expected error under a movement budget. On the synthetic 128-expert model:

```
block  0: 128 experts, 9.0% resident, movement 3.58x faster, expected error 0.02001
wrote 3072 slices: 14.26 MiB of weights plus 2.84 MiB alignment padding
weights are 0.31x the source for the same experts
precision mix: pmxq3=60 pmxq4=588 pmxq8=120
```

Hot experts keep 8 bits, cold ones drop to 3 — bits follow *access frequency and
tier cost*, not structural role. The nearest prior art,
[APEX-Quant](https://github.com/localai-org/apex-quant), allocates by role
(shared vs routed) and layer position and uses no runtime traces.

Two things it deliberately will not do:

- **Routers are never requantised.** Quantisation error in a router perturbs
  expert *selection* — the "expert shift" problem — which would invalidate the
  very trace the plan came from. Routers stay in the GGUF at source precision.
- **The error budget is not a multiple of the baseline.** Once sensitivity is
  measured the baseline *is* the reference and its error is zero, so any multiple
  of zero permits nothing: the allocator would refuse every demotion and still
  report success. `--error-bits 4.5` instead means "no worse than storing
  everything at 4.5 bits", which is how the decision is actually reasoned about.

### Router lookahead

Prefetching needs to know which experts a layer will pick *before* its router has
run. These predictors are training-free — they use only routing history a running
engine already has:

| predictor | recall @ 1× top-k | @ 2× | @ 4× |
|---|---|---|---|
| frequency (baseline) | 0.100 | 0.199 | 0.386 |
| sticky | 0.415 | 0.483 | 0.604 |
| markov | 0.329 | 0.479 | 0.593 |
| **sticky+markov** | **0.415** | **0.629** | **0.739** |

*(chance is 0.100; synthetic 64-expert trace with planted temporal structure.)*

0.739 at 4× budget is roughly what PILOT-style single-layer lookahead reports
(71.6%, improved to 76.7% by folding the shared expert into the residual first),
with no trained head. Trained pre-attention routers do considerably better —
93.0% on DeepSeek-V2-Lite, 94.7% on Qwen3-30B, 97.6% on Phi-mini-MoE — and it
would be dishonest to imply otherwise. `potatomaxx predict` scores whichever you
use on *your* trace.

### The expert cache, and where cost-awareness stops helping

Expert caches almost universally use LRU. Measured against the offline optimum on
skewed routing with uniform cost:

| policy | hit rate | % of optimum | fetch time |
|---|---|---|---|
| LRU | 0.508 | 80.9% | 9834s |
| LFU | 0.493 | 78.4% | 10140s |
| **GDSF** | **0.570** | **90.7%** | **8598s** |

Note LFU is *worse* than LRU — unaged frequency counts let an early-hot entry
ossify, and GDSF's inflation term is what fixes that.

But the honest limitation, found by measurement: **GDSF's advantage disappears
when fetch cost is proportional to size.** Its key contains `cost / size`, so at a
fixed bandwidth (`cost = bytes / rate`) that term is constant, cancels, and GDSF
collapses to LFU. Cost-awareness pays only where cost *per byte* varies — across
storage tiers (a disk byte costs ~12× a RAM byte here), or via the small-request
penalty the bandwidth surface shows. Where neither holds, LRU is the better
default and `bench --compare` will show it.

### Replay: what the fetch path actually delivers

```
--- prefetch budget sweep (gdsf, sticky+markov) ---
 budget   hit rate  bytes/token      useful     tok/s
   none      0.284       693333        0.0%    108.70
      8      0.488      1032022       45.3%    128.93
     16      0.719      2216616       36.1%    125.18
     32      0.749      4131158       18.8%     82.75
```

**Prefetching is not free**, and this is the number that says so: every prediction
is a real read, charged whether the router wants it or not. Recall keeps rising
with budget while throughput peaks at roughly `top_k` and then falls, because past
that point the extra bandwidth outruns what queue depth wins back. Anyone
reporting prefetch recall without reporting the bandwidth it cost is reporting
half the result.

In the same run, on-demand fetching with LRU reached 140.66 tok/s — beating every
prefetch configuration. That is a real result on this configuration, not a bug,
and it is why `bench --compare` sweeps rather than asserting a winner.

## What is measured, and what is not

Being clear about this matters more than the headline number.

| Claim | Status |
|---|---|
| Bandwidth surface in the table above | **Measured** on the development machine |
| `verify` proves byte-identical weights | **Measured** — enforced by tests, incl. a tamper control |
| 2.00× on the synthetic fine-grained case | **Computed** from the measured surface + a synthetic trace |
| Speedup on your model | **Unknown until you run `analyse`** — that is what it is for |
| End-to-end tokens/sec improvement | **Not measured.** Needs a runtime; out of scope |
| Per-expert requantisation from measured error | **Implemented.** `build-store`; 131 tests |
| Expert-fetch throughput (`bench`) | **Computed** from the measured surface + a replayed trace |
| Predictor recall | **Measured** on the given trace, held out from fitting |
| End-to-end generated tokens/sec | **Not measured.** Needs attention, KV cache and sampling |

What is still missing is the transformer itself: attention, KV cache, sampling and
a tokeniser. `bench` moves expert weights and accounts for them exactly, but it
does not generate text, so it reports a *ceiling* on decode rate rather than a
decode rate. On a memory-bound machine that ceiling is the binding constraint,
which is why it was built first.

The quality side is measured but not evaluated: `build-store` uses real
per-expert round-trip error, which is a great deal better than the analytic proxy
it replaced, but round-trip RMSE is not perplexity. A precision plan that looks
cheap by RMSE still needs an eval before anyone trusts it in production.

## Prior art, and what is different here

Disk-resident MoE inference is an active area, and the runtime side is well
covered:

- **[Colibri](https://github.com/JustVugg/colibri)** — pure C, expert streaming,
  router-lookahead prefetch, learned hot-expert pinning, speculative decoding.
  The mature option; if you want a runtime, start there.
- **[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)** —
  sparsity-aware expert cache.
- **llama.cpp [#25294](https://github.com/ggml-org/llama.cpp/pull/25294)** —
  SSD-backed expert streaming with `O_DIRECT` and a slot cache.
- **[Oracle-MoE](https://openreview.net/forum?id=wn6WHREK9k)**, **Sticky
  Routing**, **ReMoE** — change *routing* to improve locality, at training time.

`potatomaxx` is not another runtime. It changes only the **byte layout of the
file**, which means it composes with all of the above rather than competing, and
risks nothing: the weights are provably unchanged. The routing-locality papers
modify the model; this does not.

## Why Rust

Model files are untrusted input downloaded from public hubs, and the parser is
the part of an inference stack with **no performance requirement at all**. The
recent history of this format is a run of memory-safety failures in exactly that
code path:

- **CVE-2026-27940** — integer overflow in llama.cpp's `gguf_init_from_file_impl()`
  producing an undersized heap allocation, then a 528+ byte controlled overflow.
  Itself a bypass of the fix for CVE-2025-53630.
- **CVE-2026-7482** ("Bleeding Llama", CVSS 9.1) — out-of-bounds read from
  inflated tensor dimensions in Ollama's loader, leaking process memory.

In safe Rust the first is a checked-arithmetic error and the second a bounds
check. `potatomaxx` is *more* exposed than a plain loader, because it rewrites
model files. So:

- Every crate is `#![forbid(unsafe_code)]` **except `pmx-probe`** (page-aligned
  buffers for `O_DIRECT`) and **`pmx-kernels`** (SIMD intrinsics). Each unsafe
  block carries a written invariant; the scalar kernel is authoritative and every
  vector path is tested against it.
- Every length and offset read from a file is validated against the real file
  size, with checked arithmetic, before it is trusted. Malformed input yields a
  typed error — never a panic, never a bad read.
- **Zero dependencies.** The entire workspace is `std` only, including the GGUF
  parser, the CLI, and the JSON reader.

## Layout

| Crate | Responsibility | `unsafe` |
|---|---|---|
| `pmx-gguf` | GGUF read, offset rewriting, permutation, verification | forbidden |
| `pmx-probe` | Device bandwidth surface (blob size × queue depth) | audited |
| `pmx-trace` | Trace format, co-activation statistics, synthetic traces | forbidden |
| `pmx-partition` | Expert-order optimisation against the measured surface | forbidden |
| `pmx-plan` | Residency and bit allocation by frequency × tier cost | forbidden |
| `pmx-kernels` | GGUF dequantisation, native block formats, SIMD int8 dot | audited |
| `pmx-store` | Native store: per-expert precision, contiguous experts | forbidden |
| `pmx-cache` | Expert residency cache: LRU, LFU, GDSF | forbidden |
| `pmx-predict` | Router lookahead, training-free | forbidden |
| `pmx-runtime` | Replay harness tying prefetch, cache and precision together | forbidden |
| `pmx-cli` | The `potatomaxx` binary | forbidden |

## Status

Early. Everything above runs end to end with 131 tests, and the correctness claim
— that a repack preserves weights exactly — is enforced by tests including a
tamper control. But it has been exercised on synthetic checkpoints and one laptop.

Bugs the test suite caught during development, as a sense of what is and is not
settled: a factor-of-two error in half-precision subnormal decode (found by
sweeping all 65 536 bit patterns); a store index whose writer and reader
disagreed by four bytes per record; non-deterministic cache eviction from
`HashMap` iteration order; 25 MiB of alignment padding around 6 MiB of weights;
and a `NaN` sensitivity — from a single non-finite weight — that silently
disabled precision allocation while still reporting success.

If you run it against a real MoE checkpoint, the `analyse` and `bench --compare`
output is the interesting part, especially where it says the gain is not worth it.

`docs/design.html` holds the research this came out of, including the parts that
did not survive measurement.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).
