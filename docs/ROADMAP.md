# What to build next, and why

A survey of current MoE-inference research against this project's actual state.
Ordered by value, with the reasoning and the citations, because "add a feature"
without evidence is how the layout compiler happened.

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

## 6. Rotation before quantisation, if pushing below 3 bits

**Status: worth considering, not urgent.**

Rotation-based PTQ ([QuaRot](https://arxiv.org/abs/2404.00456) and successors)
applies Hadamard rotations to suppress outliers before quantising, and is now
standard. Reported gains are largest exactly where this project's aggressive tier
would operate: around **1.5% average improvement in the 2-bit setting**.

The native block formats here are deliberately simple, and that was the right
first choice — the point was to test bit *allocation*, not to compete on
quantisers. But if the cold tier is to go to 2–3 bits on real models, a rotation
step is the cheapest available quality recovery.

## 7. The things deliberately not on this list

- **An inference engine.** Attention, KV cache and a sampler would give
  end-to-end tokens/sec, and would duplicate mature work. See
  [KERNEL.md](KERNEL.md) for the same argument applied to the kernel.
- **A GPU backend.** The target is machines without one.
- **Layout improvements.** The transformation works correctly and has no
  workload. Optimising it further would be optimising nothing.
