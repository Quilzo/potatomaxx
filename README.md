# potatomaxx

**A layout compiler for disk-resident mixture-of-experts models.**

`potatomaxx` reads a GGUF MoE checkpoint and a routing trace, works out an expert
order that turns scattered slice reads into contiguous runs, and rewrites the
file. The output is a **drop-in GGUF** — same tensor names, same shapes,
byte-identical weights, different offsets — so whatever engine you already use
reads it unchanged and gets faster.

It also tells you, honestly, when there is nothing to gain.

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

## What is measured, and what is not

Being clear about this matters more than the headline number.

| Claim | Status |
|---|---|
| Bandwidth surface in the table above | **Measured** on the development machine |
| `verify` proves byte-identical weights | **Measured** — enforced by tests, incl. a tamper control |
| 2.00× on the synthetic fine-grained case | **Computed** from the measured surface + a synthetic trace |
| Speedup on your model | **Unknown until you run `analyse`** — that is what it is for |
| End-to-end tokens/sec improvement | **Not measured.** Needs a runtime; out of scope |
| Requantisation by access frequency | **Not implemented.** The allocator exists; the kernels do not |

The bit-allocation planner (`pmx-plan`) solves a genuinely different problem from
ordinary mixed-precision quantisation — it allocates bits by
`frequency × tier cost` rather than by sensitivity alone, because the binding
constraint is bytes *moved through a tier*, not bytes *stored*. It reports what
that would buy. It does **not** requantise: that needs dequantise/requantise
kernels and a real quality evaluation, and shipping the allocator without them
would be shipping a number nobody had checked. Its `delta_loss` term is a
documented proxy, replaceable via the API.

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

- Every crate is `#![forbid(unsafe_code)]` **except `pmx-probe`**, which needs
  page-aligned buffers for `O_DIRECT`. Its unsafe blocks each carry a written
  invariant and are covered by an alignment test.
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
| `pmx-cli` | The `potatomaxx` binary | forbidden |

## Status

Early. The pipeline runs end to end and the correctness claim is enforced by
tests, but it has been exercised on synthetic checkpoints and one laptop. If you
run it against a real MoE checkpoint, the `analyse` output is the interesting
part — especially if it says the gain is not worth it.

`docs/design.html` holds the research this came out of, including the parts that
did not survive measurement.

## Licence

Apache-2.0 OR MIT, at your option.
