<!-- GitHub Discussion for ggml-org/llama.cpp — category: Show and tell -->

# potatomaxx: reorder MoE experts on disk so llama.cpp reads fewer, larger blocks

A tool that repacks a GGUF MoE checkpoint so experts that fire together sit
adjacent on disk. Output is a **drop-in GGUF** — same tensor names, same shapes,
byte-identical weights — so llama.cpp reads it unchanged. Nothing to patch.

**https://github.com/Quilzo/potatomaxx** — Rust, zero dependencies,
GPL-2.0-or-later.

I'm posting partly to share it and partly because I'd value a sanity check on the
correctness argument from people who know this format far better than I do.

## Why bother

A big MoE checkpoint is streamed from storage while it runs, and storage cares
enormously how you ask. On my laptop (i5-1235U, 7.6 GB RAM, no GPU) the same NVMe
delivers **0.099 GB/s at queue depth 1 and 3.291 GB/s at depth 16**. A factor of
33, from nothing but concurrency.

A GGUF MoE layer stacks its experts into a few tensors with the expert index as
the last axis, so fetching top-k means k slices out of each — `k × n_tensors`
scattered reads. If the experts a token wants are adjacent, those become a few
contiguous runs instead.

## The correctness argument, which is what I'd like checked

Permuting the expert axis is a **relabelling**. If new slot `j` holds old expert
`perm[j]`, and you permute the router's weight rows by the same `perm`, then the
logit for slot `j` is exactly the logit the original model computed for expert
`perm[j]`. Top-k selects the same real experts, from the same bytes.

The file-level half relies on GGUF locating tensor data solely through the
`offset` in its `tensor_info`, with physical order unconstrained and inter-tensor
padding explicitly allowed. That's my reading of the spec — **if that's wrong,
I'd rather hear it now.**

`potatomaxx verify` proves it after the fact by reading both files in full and
comparing every expert slice. It's strict enough to reject a single flipped byte,
and CI asserts that it does:

```
$ potatomaxx verify --model orig.gguf --repacked packed.gguf --plan p.pmxplan
  13 tensors compared, 1 byte-identical, 12 matched as a permutation
  12.10 MiB of weights confirmed unchanged
OK — the repacked file holds exactly the original weights.
```

Every tensor carrying a layer's expert axis is permuted together, including the
router. **Routers are never requantised** — quantisation error there perturbs
expert *selection* (the "expert shift" problem), which would invalidate the very
routing trace the plan was derived from.

## Honest scope: layout is the *secondary* lever

I originally built this around layout and the measurements disagreed. Once a
layer's top-k reads are issued concurrently — which #25294's expert streaming
does — reordering buys **1.1–1.6×**, and only for slices in a particular size
band. Above ~256 KiB slices the bandwidth surface plateaus and layout barely
matters.

So this is not a big win for Mixtral-shaped models with large experts. It's for
fine-grained MoE — hundreds of small experts per layer — where slices land in the
16–128 KiB range where request size still matters a lot. `potatomaxx analyse`
tells you which case you're in, and says plainly when the answer is "leave it
alone":

```
 layer   experts    req/token        after   speedup  verdict
     0       128        23.37         7.48     3.01x  repack
```

## It needs a routing trace

The whole thing keys off which experts *your* workload actually selects. The text
format is one line per `(token, layer)`:

```
0 0 12 44 7 91
0 1 3 55 8 12
```

**This is the ask.** A small patch to dump router top-k would make this usable
against real models, and would be independently useful for anyone studying expert
locality. I'd be glad to write it if there's interest in the shape it should
take — or to be told the routing data is already reachable somewhere I've missed.

## Also possibly of interest: two I/O findings

From `potatomaxx kio`, which compares read paths on your device. Both are relevant
to #25294:

- **How you drive io_uring matters more than using it.** Submitting a batch and
  waiting for all of it drains the ring between batches; that peaked at QD8 and
  then *declined*. A sliding window — replace each completed read immediately —
  gained **+148% at QD8**.
- **`RWF_DONTCACHE` (6.14) measured 49% slower and is worth having anyway.** It
  left 7.4% page-cache residency instead of filling RAM, and `cachestat()` shows
  the plain path thrashing. It's not a throughput optimisation; it's a decision
  not to evict everyone else's working set to hold 40 GB you'll read once.

Caveat on all of it: measured under WSL2, so the block path is virtualised, and a
thread pool actually beat my io_uring at QD16 (3.29 vs 1.54 GB/s). I'd trust the
*directions* more than the absolute numbers, and I'd welcome results from bare
metal.

## Try it without downloading anything

```bash
cargo build --release
potatomaxx synth      # synthetic MoE checkpoint + trace
potatomaxx kio        # which read path is fastest here
potatomaxx analyse --model synth.gguf --trace synth.pmxtrace --probe surf.json
```

Happy to be told the layout argument is wrong, or that the gain isn't worth the
repack for real checkpoints. That's most of why I'm posting.
