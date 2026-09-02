# What to build next, and why

A survey of current MoE-inference research against this project's actual state.
Ordered by value, with the reasoning and the citations, because "add a feature"
without evidence is how the layout compiler happened.

The strategic conclusion first, because it reframes the rest: **the differentiator
is not an optimisation, it is the willingness to say no.** Every comparable tool —
Colibri, MoE-Infinity, llama.cpp's streaming work, MC-MoE, BitsMoE — is an
optimiser that assumes its transformation helps. This project has now measured four
plausible optimisations as not worth doing, including two of its own:

| idea | verdict | evidence |
|---|---|---|
| reorder experts by co-activation | **no payoff yet** | none of 7 checkpoints inside the 256 KiB band at Q4_K; closest misses by 1.1x |
| lossless entropy coding of the store | **2%** on k-quants | nibble entropy 3.861 of 4.000 |
| LFU expert cache | **worse than LRU** | 78.4% vs 80.9% of the offline optimum |
| repack real Granite | **leave alone**, all 24 layers | 420 KiB slices, above the plateau |

Each of those would have been weeks of work for nothing. `potatomaxx advise`
exists to produce that verdict cheaply, per model and per device, and is the
feature least likely to be duplicated — because it requires being willing to
report that your own idea does not work.

Context: the layout transformation this project was built around
[helps no real model](REAL-MODEL-RESULTS.md). What survives is queue depth,
per-expert precision, prefetch prediction, cache policy, and the correctness
machinery. The items below are ranked against that.

---

## 1. Super Expert protection — a safety hole, not a feature

**Status: partially fixed. The detection half is missing.**

[Unveiling Super Experts in MoE LLMs](https://arxiv.org/abs/2507.23279) reports
that **under 0.5% of experts** — one to ten in an entire model; 3 of 6,144 in
Qwen3-30B-A3B, 10 of 15,677 in DeepSeek-R1 — produce extreme activation outliers
in their `down_proj` output. Those propagate through the residual stream and
sustain the model's attention sinks. Pruning them costs **21–27% average accuracy,
52–74% on GSM8K**, and collapses reasoning models to near zero Pass@1, with a
90%+ attention-sink decay rate.

This is a direct hazard to what `pmx-plan` does. Allocating bits by access
frequency will quantise a *cold* expert hardest, and a Super Expert can be cold.
Worse, the error budget cannot catch it: the round-trip RMSE of that expert's own
weights is entirely ordinary, because the damage is in the activations it
produces, not in its weights.

Mitigated so far:

- `PlanConfig::protected_experts` — never demoted.
- `PlanConfig::require_protection_list`, **default true** — with no analysis
  supplied, nothing is demoted at all. The safe failure mode for an unverified
  model is to change nothing.
- `PlanConfig::floor_bits` — a hard floor regardless of budget.
- `build-store` prints the hazard when it proceeds without a list.

What is missing is detection, and the paper is clear that **weight magnitude is
not sufficient** — it must come from activations. The good news is it needs
[a single input prompt](https://arxiv.org/html/2411.07191v1), not a validation
set, and Super Experts are model-specific and stable across post-training. So one
forward pass with tensor capture identifies them permanently for a given model.

Which is the same mechanism item 2 needs.

## 2. A real routing trace, via `llama.cpp`'s eval callback

**Status: not started. Highest-value unlock.**

Every trace this project has used is synthetic. That is the single largest gap: it
means cache hit rates, prefetch recall and residency plans are exercised but not
validated against a real access pattern.

The fix does not need a fork. `llama.cpp` ships
[`examples/eval-callback`](https://github.com/ggml-org/llama.cpp/tree/master/examples/eval-callback),
which uses `ggml_backend_sched_eval_callback` to observe every intermediate tensor
during inference with its name, shape and data — and others have used it for
expert inspection without patching the tree. One harness built on it yields:

- **router top-k per token per layer** → the real routing trace, and
- **`down_proj` output magnitudes** → the Super Expert list from item 1.

Both problems, one tool. This is what to build next.

## 3. Contribute the cache-policy measurements upstream

**Status: ready to send. Low effort, high value.**

`llama.cpp` [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
requests a two-tier GPU+RAM expert cache with pluggable eviction, and is
explicitly looking for contributors. It proposes SLRU as the default, notes LRU's
prefill-wipes-the-cache failure mode, and offers **LFU** among the options. It
reports 1.9–2.5 tok/s cold rising to 12–14 tok/s at 98–100% hit rate.

`pmx-cache` has measurements that bear directly on those choices, and one
contradicts them:

- **LFU is worse than LRU** on skewed routing — 78.4% of the offline optimum
  against 80.9%. Unaged frequency counts ossify: an entry hot early keeps its
  count for ever. Offering LFU as a policy option invites a regression.
- **GDSF reaches 90.7%** of the offline optimum, beating both, because its
  inflation term is exactly the ageing LFU lacks. Its frequency-gated behaviour is
  also what #20757 asks for under "experts enter Tier 1 on their second miss".
- **Hit rate is the wrong objective for a tiered cache.** With mixed costs, GDSF
  achieves a *lower* hit rate than LFU and still spends less time fetching. A
  three-tier VRAM/RAM/SSD design is precisely where that bites, and #20757 is
  currently tuning on hit rate.
- **Measure against the offline optimum**, not against other policies. It is
  computable from a trace and turns "policy A beat policy B" into "policy A
  captured 90.7% of what was available".

## 4. Model the prefill/decode phase split

**Status: not started. Cheap, and needed for item 3 to be honest.**

#20757's central observation is that a prefill burst wipes an LRU cache and
starves the decode that follows. `pmx-trace` cannot express that: its synthetic
traces are uniform, so they cannot distinguish SLRU from LRU at all, and the
policy comparison in item 3 is therefore blind to the failure mode the upstream
issue is actually about.

Adding a prefill phase — many tokens, wide expert coverage — followed by decode
would let the comparison speak to it. This is a small change to `SynthConfig`,
and it is the difference between a measurement that is relevant upstream and one
that is not.

## 5. Correct the novelty claim on frequency-based bit allocation

**Status: needs doing. Honesty issue.**

This project has claimed that nothing allocates bit width per expert from
observed access frequency. That is **wrong**, and the GNU submission repeats it.

[MC-MoE](https://arxiv.org/abs/2410.06270) (ICLR 2025) formulates adaptive
bit-width allocation as a linear program over per-expert importance, explicitly
including **activated frequency** alongside reconstruction error and routing
scores, reaching 2.54 bits with 3.8% accuracy loss.
[MoQa](https://arxiv.org/html/2606.17118v1) extends it with channel-level
adjustment; [BitsMoE](https://arxiv.org/html/2606.00079v1) allocates by spectral
energy instead.

What remains distinct here is narrower and should be stated as such: this
optimises **bytes moved through a storage tier** — a disk byte costing ~12× a RAM
byte on the measured device — rather than accuracy per byte *stored*, and drives
it from the operator's own runtime traces rather than a calibration set. That is
a real difference in objective, not a new idea about frequency.

## 6. Lossless entropy coding — measured, and it does not pay here

**Status: investigated and rejected for quantised stores. Kept because the
negative result is worth having.**

The literature is encouraging at first glance:
[EntroLLM](https://arxiv.org/html/2505.02380) reports Huffman coding taking 4-bit
weights to 1.39 effective bits with a 146% generation-latency improvement;
[DFloat11](https://arxiv.org/pdf/2504.11651) gets 70% size at 100% accuracy; ZipNN
takes FP16 from 16 to ~11 bits per parameter. And the objection usually raised —
that entropy decoding is serial and ill-suited to GPU SIMT, and that decompressing
to memory before the kernel erodes the bandwidth saving — is a *GPU* objection.
This project is CPU-only with idle cores and controls its own read path, so
trading spare compute for scarce disk bandwidth is exactly the right trade here.

So it was measured, on real Granite weights:

| source | zlib | lzma | effective bits | saving |
|---|---|---|---|---|
| F16 | 0.773 | 0.719 | 16 → 11.50 | **28.1%** |
| Q4_K | 0.979 | 0.981 | 4.5 → 4.41 | **2.1%** |

Nibble entropy in real Q4_K is **3.861 bits against 4.000 uniform** — 3.5%
redundancy, and no coder can beat the entropy.

The reason is structural, and worth stating because it generalises: **a good
quantiser has already spent that redundancy.** K-quants fit an asymmetric
scale and minimum per 32-element sub-block, which is precisely a transform that
drives the residual distribution toward uniform. The large published gains are on
*float* weights, where the exponent field is highly redundant — and the F16 column
above reproduces ZipNN's 16→11 bpp independently, which is a useful check that the
measurement method is sound.

Verdict: not worth a decode path for a k-quant store. Worth roughly 28% for
someone streaming an F16 checkpoint, which is a real but narrow case.
`potatomaxx advise` now measures this per model and says which case you are in.

## 6b. Speculative decoding — the largest lossless lever, now costed

**Status: estimator implemented in `advise`. The runtime half is not, and belongs
to an engine.**

This was sketched in the project's first design as a "bandwidth amplifier" and
never validated. It validates, and it is now published prior art:
[MoE-SpeQ](https://arxiv.org/pdf/2511.14102) combines speculative decoding with
proactive expert prefetching and offloading — precisely the combination sketched
here. [Spec-MoEOff and SP-MoE](https://www.spheron.network/blog/speculative-decoding-moe-models-gpu-cloud/)
use it specifically to amortise expert-loading latency in memory-constrained
offloading, and Cohere report that
[MoE models get *more* from speculative decoding](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
than dense ones.

The mechanism, stated precisely: verifying a block of drafted tokens in one pass
reads the **union** of the experts those tokens need, not the sum. Adjacent tokens
reuse experts, so the union grows sublinearly, and bytes per accepted token falls.

That is measurable from a trace **without running a model**, which is why `advise`
can cost it. `Trace::mean_window_union` computes the union over non-overlapping
windows directly; expected accepted tokens follow the standard
`(1 - a^(k+1)) / (1 - a)`. On a synthetic trace with 0.85 persistence:

| acceptance | best depth | experts per accepted token | gain |
|---|---|---|---|
| 0.50 | 2 | 7.2 | 1.10x |
| 0.70 | 2 | 5.8 | 1.38x |
| 0.90 | 8 | 4.3 | 1.85x |

Two things worth noting. The optimal draft depth **rises with acceptance** — a
deep draft wastes reads on rejected tokens when acceptance is low — which is
emergent from the model rather than assumed, and is pinned by a test.

And the gain is smaller than the literature's headline. The 2–4x usually quoted is
*throughput on GPUs* and includes compute-side wins that do not exist on a
bandwidth-bound machine. Bytes per accepted token is the honest quantity here, and
1.1–1.9x is what it gives. It is nonetheless the largest **lossless** lever
available: every precision lever trades quality, and this one does not.

## 6c. Expert pruning — bigger than quantisation, and riskier

**Status: headroom reported in `advise`. Doing it needs a saliency measure.**

Removing an expert removes its bytes entirely, which strictly beats quantising it.
The literature is clear that pruning beats *merging*
([REAP](https://arxiv.org/html/2510.13999)): merging "introduces irreducible
errors by eliminating the router's ability to maintain fine-grained, independent
control", while pruning preserves routing independence and wins at high
compression. Training-free one-shot methods exist — REAP scores by router
gate-values and expert activation norms, EASY-EP by gate-weighted output norms,
AIMER is calibration-free.

Note what those signals are **not**: frequency. Published methods deliberately
score by output magnitude rather than selection count, for the same reason the
outlier-expert hazard exists — a rarely-selected expert can be load-bearing.

So `advise` reports only the strictly-evidenced part: experts that were **never
selected** across the trace, as an upper bound on what could be safe to drop for
that workload, with the caveat attached. Anything beyond that needs activations,
and arrives with item 2.

## 7. Route-flip measurement after requantisation

**Status: not started. Needs the harness from item 2. Genuinely differentiated.**

Requantising experts changes the hidden states they produce, which changes routing
at later layers. That is the "expert shift" problem, and it is measurable.

[Causal Route-Mediated Damage in Quantized MoE](https://arxiv.org/html/2608.11212)
quantifies it on OLMoE-1B-7B: a **route-mediated fraction of 0.31** — about a third
of quantisation damage flows through routing changes rather than through arithmetic
error — reproduced across five runs at 0.313 ± 0.020, with 99.8% of net signed
damage associating with route-set changes.

Two things follow, and the second is the interesting one.

Route flips are **detectable**: a router-margin statistic reaches AUC 0.772. But
whether a given flip *helps or hurts* is **not** predictable — every predictor
class they tried, including cross-layer router vectors over all 16 layers, scored
AUC 0.490, which is chance.

So the honest feature is a **risk indicator, not a damage predictor**: "your
requantised store flips N% of routing decisions; roughly a third of quantisation
damage typically travels this path, and no published method can tell which flips
are harmful." That is more useful than it sounds — it is a number nobody currently
reports, it correlates with a real damage channel, and stating its limitation is
what keeps it honest. `VSRAQ` and `EAQuant` optimise *for* routing consistency;
none of them tell an operator what their own store did.

Needs a forward pass, so it arrives with item 2.

## 8. Rotation before quantisation, if pushing below 3 bits

**Status: worth considering, not urgent.**

Rotation-based PTQ ([QuaRot](https://arxiv.org/abs/2404.00456) and successors)
applies Hadamard rotations to suppress outliers before quantising, and is now
standard. Reported gains are largest exactly where this project's aggressive tier
would operate: around **1.5% average improvement in the 2-bit setting**.

The native block formats here are deliberately simple, and that was the right
first choice — the point was to test bit *allocation*, not to compete on
quantisers. But if the cold tier is to go to 2–3 bits on real models, a rotation
step is the cheapest available quality recovery.

## 9. Activation sparsity — does not transfer to streaming

**Status: rejected, with a reason derived from this project's own measurements.**

The most-cited remaining idea. Contextual sparsity leaves 80–90% of FFN neurons
unused per token in ReLU-style networks, and DejaVu-class predictors reach ~93%
accuracy; PowerInfer exploits it with hot/cold neuron placement.

It does not help here, and the reason is quantitative. Acting on neuron-level
sparsity means reading individual rows of an expert matrix. For Granite that is
about **342 bytes**. The measured floor on this device is **0.02 GB/s at 4 KiB**,
and useful bandwidth begins around 256 KiB — three orders of magnitude larger than
a row.

The general form is worth stating because it is easy to get wrong: **activation
sparsity saves compute on weights already in memory. It does not save bytes when
streaming from storage**, because the read is what you are trying to avoid and you
must issue it to discover the values. `advise` reports this per model.

## 10. The things deliberately not on this list

- **An inference engine.** Attention, KV cache and a sampler would give
  end-to-end tokens/sec, and would duplicate mature work. See
  [KERNEL.md](KERNEL.md) for the same argument applied to the kernel.
- **A GPU backend.** The target is machines without one.
- **Layout improvements.** The transformation works correctly and has no
  workload. Optimising it further would be optimising nothing.
