<!-- r/LocalLLaMA post -->

**Title:** I measured why MoE models are slow off SSD: queue depth is worth 33x, and most of the obvious fixes aren't the win

---

I have an 8 GB laptop with no GPU and wanted to understand *why* big MoE models
crawl off disk, rather than just accepting it. Spent a while measuring instead of
guessing, and several things I expected turned out to be wrong. Sharing both.

Tool is here, MIT-compatible-ish (GPL-2.0-or-later), zero dependencies, runs on a
synthetic model so you can try it in 30 seconds without downloading anything:
**https://github.com/Quilzo/potatomaxx**

## The headline

Same NVMe, same 64 KiB reads, 18 GiB file so the page cache can't cheat:

| queue depth | GB/s |
|---|---|
| 1 | 0.099 |
| 4 | 0.537 |
| 8 | 1.522 |
| 16 | **3.291** |

**33x, from concurrency alone.** Nothing about the drive changed. If your engine
fetches experts one at a time and waits, that's the whole story — and it's why
prefetch *prediction* matters: you can't queue reads you haven't predicted.

## Things I got wrong

I'm listing these because "here's my cool optimisation" posts are cheap and the
failures are more useful.

**Reordering experts on disk is the secondary lever, not the primary one.** I
built the tool around this. Then measured: once you issue a layer's top-k reads
concurrently, reordering buys 1.1–1.6x, and only when expert slices are in the
16–128 KiB band. Fine-grained MoE (hundreds of small experts) benefits; Mixtral-
shaped models with big experts basically don't.

**Fixed 2 MiB "groups" you always read whole are a net loss.** The over-read costs
more than the coalescing saves once slices are over ~256 KiB.

**Prefetching is not free, and more prefetch is not better.** Every prediction is
a real read. Recall keeps climbing with budget, but *throughput peaks at roughly
top_k and then falls*. In one config, plain on-demand LRU beat every prefetch
setting I tried. Anyone quoting prefetch hit rates without the bandwidth they
cost is quoting half the result.

**LRU is worse than you'd think, but the "smart" policy has a catch.** A
frequency-and-cost-aware policy (GDSF) hit 90.7% of the theoretical optimum vs
LRU's 80.9% on skewed expert routing. *But* its advantage vanishes when fetch cost
is proportional to size — which it is at fixed bandwidth — because then the cost
term cancels and it degenerates into plain LFU, which is *worse* than LRU.
Cost-awareness only pays across storage tiers.

**`RWF_DONTCACHE` was 49% slower — and I'd still use it.** New-ish kernel flag
(6.14) that reads through the page cache but drops the pages after. Slower for
*me*, but it left 7.4% page-cache residency instead of filling RAM. It's not a
speed knob, it's a "don't evict everything else on the machine to cache 40 GB
you'll read once" knob. `cachestat()` confirms the plain path thrashes.

**My io_uring lost to 16 threads.** 1.54 vs 3.29 GB/s. And my *first* io_uring
version was much worse still — I submitted a batch and waited for all of it, which
drains the ring between batches; switching to a sliding window gained 148% at
QD8. Caveat: this is WSL2, so virtualised block path, and I don't have a bare-metal
box to check on. `potatomaxx kio` reproduces the whole table in about a minute if
someone wants to tell me I'm wrong.

## What the tool does

Three things, and it measures each rather than claiming it:

1. **Repacks a GGUF** so co-firing experts are adjacent. Output is a drop-in
   GGUF — byte-identical weights, so llama.cpp reads it unchanged. There's a
   `verify` command that proves the weights didn't change and rejects a single
   flipped byte.
2. **Per-expert precision.** Hot experts at 8 bits, cold ones at 3, chosen from
   *measured* quantisation error per expert plus how often it's actually used.
   3.58x less weight movement at 0.31x the size on my synthetic model. (Can't be
   done in GGUF — a tensor has one type — so it writes its own format for this.)
3. **Router lookahead.** Training-free predictors hit 0.739 recall at 4x budget
   vs 0.100 chance. Roughly what published single-layer lookahead reports, with no
   trained head.

## Big honest caveat

**It has never been run against a real MoE checkpoint.** Everything above is
synthetic models plus real device measurements on one laptop. It needs a routing
trace (which experts your workload picks) and getting that out of a real engine
needs a small patch.

So if you have a Qwen3-30B-A3B or similar and 20 minutes, the `analyse` output
would be genuinely useful to me — *especially* if it says the gain isn't worth it.
A tool whose job is telling you whether an optimisation helps is worthless if it
can't be trusted when it says no.

```bash
cargo build --release
./target/release/potatomaxx synth   # no download needed
./target/release/potatomaxx kio     # measure your own drive
```
