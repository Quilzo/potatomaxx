# Results on a real MoE checkpoint

Until now every number in this repository came from synthetic checkpoints, and
the README said so. This records the first run against a real model, including
the parts where the tool declined to claim a win.

## The model

[`bartowski/granite-3.0-1b-a400m-instruct-GGUF`](https://huggingface.co/bartowski/granite-3.0-1b-a400m-instruct-GGUF),
`Q4_K_M`, 822 MB. Chosen because it is a genuine mixture-of-experts model small
enough to iterate on: `granitemoe`, 24 MoE blocks, 32 experts per layer, top-8
routing, `n_embd` 1024, `n_ff` 512.

Ground truth for the decoder validation is the `f16` build of the same model from
the same repository.

## Structure detection

```
$ potatomaxx inspect granite-moe-Q4_K_M.gguf
  architecture granitemoe
  tensors      242
  data section 782.12 MiB

24 MoE layers
  experts/layer      32
  slice (largest)    420.00 KiB
  bytes/expert       996.00 KiB
  precision          Q6_K (5.19 bits/weight)
  expert-axis tensors per layer: 4
    blk.0.ffn_gate_exps.weight
    blk.0.ffn_up_exps.weight
    blk.0.ffn_down_exps.weight
    blk.0.ffn_gate_inp.weight
  expert weight total: 697.50 MiB
```

Detection worked on the real tensor naming without changes, and picked up the
router (`ffn_gate_inp`) as carrying the expert axis — which matters, because the
router must be permuted with the experts or the model computes something else.

Worth noting independently: **697.5 of 782 MiB — 89% of the checkpoint — is
routed expert weight.** That is the premise of the whole project, on a real file.

The type mix is also more varied than the synthetic model exercised: 144 Q4_K, 25
Q6_K and 73 F32 tensors overall; of the expert tensors, 60 Q4_K and 12 Q6_K.

## The correctness claim, on real weights

This is the result that matters most, because it is the one thing the project
cannot be wrong about. A deterministic shuffle was applied to all 32 experts in
all 24 layers — 96 tensors, real Q4_K and Q6_K slices — and the output compared
against the source in full.

```
$ potatomaxx pack --model granite-moe-Q4_K_M.gguf --plan granite.pmxplan --out granite.packed.gguf
  24 layers, 96 tensors permuted
  wrote 782.12 MiB of tensor data (8 B padding)
real 3.4s

$ potatomaxx verify --model granite-moe-Q4_K_M.gguf --repacked granite.packed.gguf --plan granite.pmxplan
  242 tensors compared, 146 byte-identical, 96 matched as a permutation
  782.12 MiB of weights confirmed unchanged
OK — the repacked file holds exactly the original weights.
real 3.0s
```

File size identical, 821,845,024 bytes in and out.

A shuffle rather than a reversal on purpose: it forces non-adjacent slice moves,
which is harder on the code.

**Tamper control.** Four bytes flipped at the midpoint of the repacked file:

```
error: VERIFICATION FAILED: tensor "blk.19.ffn_up_exps.weight" slice 12
       does not match source slice 19
```

Rejected, and localised to the exact tensor and slice.

## Dequantiser validation against ground truth

The dequantisers were written from the ggml block layouts and, until now, tested
only against synthetic blocks constructed by hand — which proves the arithmetic
matches *my reading* of the layout, precisely what a self-written test cannot
check.

Comparing the same tensor from the `f16` build against the `Q4_K_M` build fixes
that. Half-to-float is exhaustively verified over all 65,536 bit patterns, so the
F16 side is ground truth.

| tensor | reference | candidate | elements | RMSE | max abs | correlation | rel. error |
|---|---|---|---|---|---|---|---|
| `blk.0.ffn_gate_exps.weight` | F16 | Q4_K | 16,777,216 | 0.001202 | 0.01508 | **0.997089** | 7.63% |
| `blk.0.ffn_down_exps.weight` | F16 | Q6_K | 16,777,216 | 0.000327 | 0.01388 | **0.999822** | 1.89% |

Both pass, and two things corroborate them beyond the threshold:

- **Correlation is the discriminating statistic.** A layout misreading destroys
  correlation while potentially leaving the error magnitude superficially
  plausible; quantisation error does the opposite. 0.997 and 0.9998 are what
  quantisation noise looks like.
- **The two errors are in the right ratio.** Q6_K carries 6.5625 bits per weight
  against Q4_K's 4.5, and is four times more accurate here (1.89% against 7.63%).
  Independent decoders sharing no code — Q4_K packs 6-bit scales and mins with an
  irregular split across bytes, Q6_K uses 8-bit scales and a −32 bias — landing in
  the ratio their bit-widths predict is not something two coincidentally wrong
  implementations produce.

Reproduce with `potatomaxx compare --a f16.gguf --b q4km.gguf`. The comparison
needs only one tensor from each file, so an F16 reference can be range-fetched
rather than downloaded whole.

## Where the tool declined to claim a win

`analyse`, on all 24 layers:

```
 layer   experts    req/token        after   speedup  verdict
     0        32        18.37        19.50     1.02x  leave alone
     1        32        18.42        19.47     1.02x  leave alone
   ...
```

**Every layer: leave alone.** This is the correct answer and it is worth dwelling
on. Granite's expert slices are 420 KiB, well above the ~256 KiB point where the
measured bandwidth surface plateaus, so reordering cannot buy anything — the reads
are already large enough that request size has stopped mattering. The tool says so
instead of manufacturing a number.

The same applies to precision. `build-store` produced a store **0.99× the size of
the source**: granite is already Q4_K at 4.5 bits per weight, and the native Q4
format is also 4.5, so there is nothing to reclaim. Measured expected error on
real weights came out at 0.009–0.011 per block, with a precision mix of 2,121
Q4, 180 Q8 and 3 Q3 slices.

Alignment padding is a real cost that shows up honestly here: 106.75 MiB of
padding around 690 MiB of weights, about 15%, from 2 MiB group alignment across 96
groups. On this model that is a straightforward loss, which is why it is reported
as a separate figure rather than folded into the total.

`bench`, replaying a shape-matched trace against the real store:

```
  hit rate         0.635          (64 MiB cache)
  bytes/token      414.35 MiB
  prefetch useful  31.7%
  effective read   1.89 GB/s
  fetch-limited    4.34 tok/s
```

## What this does and does not establish

**Established on a real model:**

- MoE structure detection, on real tensor naming, including the router.
- The repacker preserves every weight exactly across 96 permuted real Q4_K and
  Q6_K tensors, and `verify` catches a four-byte corruption.
- Both k-quant dequantisers are correct against F16 ground truth.
- The precision allocator's error measurements run on real weights.
- The tool reports "no gain available" when that is the truth.

**Still not established:**

- **Routing is synthetic.** Real routing requires dumping a router's top-k from a
  running engine, which needs a small patch to one. The trace used here has the
  model's real *shape* — 24 layers, 32 experts, top-8 — but its selections are
  generated. So the layout and residency decisions are exercised, but not
  validated against a real access pattern. This is the single largest gap and the
  main thing being asked for upstream.
- **No end-to-end tokens per second.** There is still no attention, KV cache or
  sampler here; `bench` reports a ceiling on decode rate, not a decode rate.
- **A model where layout should help has not been tested.** Granite's slices are
  too large to benefit. The interesting case is fine-grained MoE with hundreds of
  small experts, where slices land in the 16–128 KiB band. Qwen3-30B-A3B or
  DeepSeek-shaped models are the test that would either support or sink the layout
  argument.
- **Quality is measured as round-trip error, not perplexity.** A precision plan
  that looks cheap by RMSE still needs a real evaluation.
