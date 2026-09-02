// SPDX-License-Identifier: GPL-2.0-or-later
//! `potatomaxx advise` — say which optimisations are worth attempting, and which are not.
//!
//! # Why this is the point of the project
//!
//! Every comparable tool is an *optimiser*: it assumes its transformation helps
//! and applies it. This one measures first and is willing to say no. That has
//! turned out to be the more valuable half — five times now, twice on this
//! project's own ideas and once on a proposal about to ship elsewhere:
//!
//! | idea | measured verdict |
//! |---|---|
//! | reorder experts on disk by co-activation | no real model has slices small enough; **useless** |
//! | lossless entropy coding of the store | 2% on k-quants (28% on F16); **not worth it** |
//! | LFU expert cache | *worse* than LRU on skewed routing |
//! | SLRU, the policy llama.cpp #20757 wants as its default | a 47-point *regression* on expert streaming |
//! | repack a real Granite checkpoint | leave it alone, on every layer |
//!
//! Each of those would have been weeks of work for nothing. Producing that
//! verdict cheaply, per model and per device, is what this command does.
//!
//! Nothing here is a new optimisation. It is the accumulated set of thresholds at
//! which the known optimisations stop paying, applied to your inputs.

use crate::moe;
use pmx_gguf::Gguf;
use pmx_kernels::can_dequantize;
use pmx_probe::Surface;
use pmx_trace::{CoActivation, Trace};

/// How strongly a finding argues for action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Worth doing, with a quantified gain.
    Do,
    /// Measurably not worth doing.
    Skip,
    /// Real but small. Reported separately because rounding a borderline signal
    /// down to "no" is how an advisor loses the user's trust.
    Marginal,
    /// A correctness or quality hazard, not a performance question.
    Hazard,
    /// Cannot be judged from the inputs supplied.
    Unknown,
}

impl Verdict {
    fn tag(self) -> &'static str {
        match self {
            Verdict::Do => "DO",
            Verdict::Skip => "SKIP",
            Verdict::Marginal => "MARGINAL",
            Verdict::Hazard => "HAZARD",
            Verdict::Unknown => "UNKNOWN",
        }
    }
    /// Sort order: hazards first, then wins, then unknowns, then dead ends.
    fn rank(self) -> u8 {
        match self {
            Verdict::Hazard => 0,
            Verdict::Do => 1,
            Verdict::Marginal => 2,
            Verdict::Unknown => 3,
            Verdict::Skip => 4,
        }
    }
}

struct Finding {
    verdict: Verdict,
    topic: &'static str,
    /// The number the verdict rests on.
    evidence: String,
    /// What to do, or why not to bother.
    advice: String,
}

/// The request-size threshold above which the bandwidth surface stops rewarding
/// larger reads, so coalescing cannot help.
const PLATEAU_BYTES: u64 = 256 * 1024;
/// Below this lossless saving, entropy coding is not worth the decode path.
const ENTROPY_WORTH_IT: f64 = 0.10;

/// Turn a "measured over baseline" ratio into a verdict.
///
/// The middle band exists because the first version of this had a bare 1.5x
/// threshold and reported a genuine 1.48x prefetch signal as a dead end.
fn classify(ratio: f64) -> Verdict {
    if ratio >= 1.5 {
        Verdict::Do
    } else if ratio >= 1.15 {
        Verdict::Marginal
    } else {
        Verdict::Skip
    }
}

/// Empirical entropy of a byte slice, in bits per byte.
fn entropy(bytes: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for b in bytes {
        hist[*b as usize] += 1;
    }
    let n = bytes.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    -hist
        .iter()
        .filter(|c| **c > 0)
        .map(|c| {
            let p = *c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Run every check the supplied inputs allow.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: &str,
    trace_path: Option<&str>,
    surface: &Surface,
    cache_mib: u64,
    alpha: f64,
) -> Result<(), String> {
    let g = Gguf::open(model).map_err(|e| format!("reading {model}: {e}"))?;
    let m = moe::detect(&g);
    if m.layers.is_empty() {
        return Err(format!("{model} has no MoE layers; nothing here applies"));
    }
    let layer = &m.layers[0];
    let mut f: Vec<Finding> = Vec::new();

    println!("advising on {model}");
    println!(
        "  {} MoE layers x {} experts, {} per expert\n  source {} at {:.2} bits/weight \
         (mean {:.2}; the minimum is what bounds requantisation)\n",
        m.layers.len(),
        layer.n_experts,
        human(layer.bytes_per_expert),
        layer.baseline_type,
        layer.min_baseline_bits,
        layer.baseline_bits
    );

    // --- 1. Queue depth. The largest lever, and independent of the model. ---
    if surface.cells.is_empty() {
        f.push(Finding {
            verdict: Verdict::Unknown,
            topic: "queue depth",
            evidence: "no measured bandwidth surface".into(),
            advice: "run `potatomaxx probe` first. On the development machine this was worth \
                     33x and dominated every other lever."
                .into(),
        });
    } else {
        let s = layer.slice_bytes.max(4096);
        let qd1 = surface.bandwidth_at(s, 1).unwrap_or(0.0);
        let qd16 = surface.bandwidth_at(s, 16).unwrap_or(0.0);
        let ratio = if qd1 > 0.0 { qd16 / qd1 } else { 0.0 };
        f.push(Finding {
            verdict: if ratio > 1.5 {
                Verdict::Do
            } else {
                Verdict::Skip
            },
            topic: "queue depth",
            evidence: format!(
                "{:.3} GB/s at QD1 vs {:.3} at QD16 for {} reads = {ratio:.1}x",
                qd1 / 1e9,
                qd16 / 1e9,
                human(s)
            ),
            advice: if ratio > 1.5 {
                "Keep many reads in flight. This is almost certainly the biggest win \
                 available, and it needs an engine change, not a file change."
                    .into()
            } else {
                "This device does not reward concurrency; look elsewhere.".into()
            },
        });
    }

    // --- 2. Layout. Settled by slice size, but report the distance. ---
    // An earlier version answered this yes/no and overstated the result: the
    // closest real model misses the threshold by only 10%, and moves inside it at
    // Q2_K. Distance is the useful answer, not a verdict.
    let ratio = layer.slice_bytes as f64 / PLATEAU_BYTES as f64;
    // Bits per weight at which this model's slices would enter the band.
    let weights_per_slice = (layer.slice_bytes as f64 * 8.0) / layer.min_baseline_bits;
    let enter_bits = (PLATEAU_BYTES as f64 * 8.0) / weights_per_slice;
    f.push(Finding {
        verdict: if ratio < 1.0 {
            Verdict::Do
        } else if ratio < 1.5 {
            Verdict::Marginal
        } else {
            Verdict::Skip
        },
        topic: "expert layout",
        evidence: format!(
            "slices are {} against a ~{} plateau = {ratio:.2}x outside",
            human(layer.slice_bytes),
            human(PLATEAU_BYTES)
        ),
        advice: if ratio < 1.0 {
            "Slices are inside the band where request size still matters, so coalescing \
             reads can genuinely help. Run `analyse`."
                .into()
        } else if ratio < 1.5 {
            format!(
                "Just outside. Reordering will buy little at this precision, but these \
                 slices enter the band below about {enter_bits:.1} bits/weight -- so a more \
                 aggressively quantised build of the same model may benefit. Worth an \
                 `analyse` run to check."
            )
        } else {
            format!(
                "Reads are already large enough that request size has stopped mattering, so \
                 reordering cannot help. It would take about {enter_bits:.1} bits/weight to \
                 enter the band. Note the architectural trend runs the other way: the \
                 fine-grained MoE scaling law favours more, smaller experts, so this may \
                 become relevant for future models."
            )
        },
    });

    // --- 3. Precision headroom, bounded by the source format. ---
    let src_bits = layer.min_baseline_bits;
    let headroom = src_bits - 2.5;
    f.push(Finding {
        verdict: if headroom > 1.0 {
            Verdict::Do
        } else {
            Verdict::Skip
        },
        topic: "precision",
        evidence: format!("source is {src_bits:.2} bits/weight; floor is 2.50"),
        advice: if headroom > 1.0 {
            format!(
                "Up to {:.0}% fewer bytes to move is available. Requantising cannot recover \
                 information, so the ladder is capped at the source -- an already-2-bit \
                 checkpoint has nothing left.",
                (1.0 - 2.5 / src_bits) * 100.0
            )
        } else {
            "Already at or near the floor; requantisation has nothing left to give.".into()
        },
    });

    // --- 4. Lossless entropy coding. Measured on this model's own bytes. ---
    match sample_entropy(&g, layer) {
        Some((bits_per_byte, sampled)) => {
            let saving = 1.0 - bits_per_byte / 8.0;
            f.push(Finding {
                verdict: if saving > ENTROPY_WORTH_IT {
                    Verdict::Do
                } else {
                    Verdict::Skip
                },
                topic: "lossless coding",
                evidence: format!(
                    "expert bytes carry {bits_per_byte:.2} bits of entropy per byte \
                     ({:.1}% redundant, {} sampled)",
                    saving * 100.0,
                    human(sampled)
                ),
                advice: if saving > ENTROPY_WORTH_IT {
                    format!(
                        "Entropy coding the store could cut roughly {:.0}% of bytes moved at \
                         zero quality cost. Float checkpoints have redundant exponent fields; \
                         this one does.",
                        saving * 100.0
                    )
                } else {
                    "Not worth a decode path. A good quantiser leaves its output near-uniform \
                     by construction, so there is little redundancy to reclaim -- measured at \
                     ~2% on k-quants against ~28% on F16."
                        .into()
                },
            });
        }
        None => f.push(Finding {
            verdict: Verdict::Unknown,
            topic: "lossless coding",
            evidence: "expert tensor type not decodable by this build".into(),
            advice: "cannot sample the weight bytes".into(),
        }),
    }

    // --- 4b. Activation sparsity, and why it does not transfer to streaming. ---
    // Worth stating because it is the most-cited remaining idea and the reason it
    // fails here is a direct consequence of this project's own measurements.
    let row_bytes =
        ((layer.slice_bytes as f64) / (layer.weights_per_expert.max(1) as f64).sqrt()) as u64;
    f.push(Finding {
        verdict: Verdict::Skip,
        topic: "activation sparsity",
        evidence: format!(
            "a single expert row is roughly {}; the device needs >= {} per read to be efficient",
            human(row_bytes.max(1)),
            human(PLATEAU_BYTES)
        ),
        advice: "Contextual sparsity leaves 80-90% of neurons unused per token in ReLU-style \
                 FFNs, and predictors reach ~93% accuracy. But acting on it requires reading \
                 individual rows, which are far below the size at which this device delivers \
                 useful bandwidth -- the measured floor is 0.02 GB/s at 4 KiB. Activation \
                 sparsity saves *compute* on weights already in RAM; it does not save bytes \
                 when streaming from storage, because you still pay a read to find out."
            .into(),
    });

    // --- 5. The quality hazard. Always reported. ---
    f.push(Finding {
        verdict: Verdict::Hazard,
        topic: "outlier experts",
        evidence: format!(
            "{} experts in this model; under 0.5% are typically critical",
            layer.n_experts as usize * m.layers.len()
        ),
        advice: "A handful of experts produce extreme activation outliers that sustain the \
                 model's attention sinks; pruning one costs 21-27% accuracy. Such an expert \
                 can be COLD, so frequency-based allocation would quantise it hardest, and its \
                 own round-trip error looks ordinary. Identify them from one forward pass \
                 before trusting any requantised store."
            .into(),
    });

    // --- 6 and 7. Trace-dependent: cache and prefetch headroom. ---
    match trace_path {
        None => f.push(Finding {
            verdict: Verdict::Unknown,
            topic: "cache & prefetch",
            evidence: "no routing trace supplied".into(),
            advice: "Both depend entirely on the access pattern. Capture a trace from your own \
                     workload -- routing is workload-specific, and so is the answer."
                .into(),
        }),
        Some(tp) => {
            let t = Trace::read(tp).map_err(|e| format!("reading {tp}: {e}"))?;
            let ca = CoActivation::from_trace(&t, 0);
            let resident = ((cache_mib << 20) / layer.bytes_per_expert.max(1)) as usize;
            let resident = resident.clamp(1, layer.n_experts as usize);
            let skew = ca.mass_in_top(resident);
            let uniform = resident as f64 / layer.n_experts as f64;
            f.push(Finding {
                verdict: classify(skew / uniform.max(1e-9)),
                topic: "expert cache",
                evidence: format!(
                    "{resident} of {} experts fit in {}; they take {:.0}% of selections \
                     (uniform would be {:.0}%)",
                    layer.n_experts,
                    human(cache_mib << 20),
                    skew * 100.0,
                    uniform * 100.0
                ),
                advice: format!(
                    "Skew is {:.2}x uniform. Above ~1.5x a small resident set clearly pays; \
                     near 1.0x it is doing little more than holding a random subset. When it \
                     does pay, prefer a frequency-and-cost-aware policy: LRU reached only \
                     80.9% of the offline optimum against GDSF's 90.7%, and plain LFU was \
                     *worse* than LRU. Optimise fetch time, not hit rate -- once tiers cost \
                     differently those diverge.",
                    skew / uniform.max(1e-9)
                ),
            });

            let adj = adjacent_overlap(&t);
            let chance = t.top_k as f64 / f64::from(t.n_experts);
            // --- Speculative decoding, costed from the trace's own reuse. ---
            // Verifying a block of drafted tokens in one pass reads the *union* of
            // the experts they need, not the sum. Because adjacent tokens reuse
            // experts, that union grows sublinearly, so bytes per accepted token
            // falls. This is measurable from a trace without running a model,
            // which is the only reason it can be advised on here.
            let top_k = f64::from(t.top_k);
            let mut rows: Vec<(usize, f64, f64)> = Vec::new();
            for depth in [2usize, 4, 8] {
                let union = t.mean_window_union(0, depth);
                // Expected accepted tokens per pass under per-token acceptance
                // `alpha`, plus the always-correct token the target produces.
                let accepted: f64 = (0..depth).map(|i| alpha.powi(i as i32 + 1)).sum::<f64>() + 1.0;
                let accepted = accepted.min(depth as f64 + 1.0);
                rows.push((depth, union / accepted, union / top_k));
            }
            let best = rows
                .iter()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .copied()
                .unwrap_or((1, top_k, 1.0));
            let gain = top_k / best.1.max(1e-9);
            f.push(Finding {
                verdict: classify(gain),
                topic: "speculative decoding",
                evidence: format!(
                    "at {alpha:.2} acceptance, depth {} reads {:.1} experts per accepted token \
                     against {top_k:.0} one-at-a-time = {gain:.2}x fewer",
                    best.0, best.1
                ),
                advice: format!(
                    "Verifying a drafted block reads the union of its experts, not the sum, and \
                     reuse makes that union sublinear ({:.1} experts over {} tokens here). It is \
                     *lossless*, which distinguishes it from every precision lever. Two caveats: \
                     acceptance must be measured rather than assumed, since the whole gain \
                     scales with it and the optimal draft depth moves with it too; and this is \
                     bytes per accepted token only -- the 2-4x figures usually quoted for \
                     speculative decoding are throughput on GPUs and include compute-side wins \
                     that do not apply to a bandwidth-bound machine.",
                    best.2 * top_k,
                    best.0
                ),
            });

            // --- Pruning headroom, workload-specific and strictly evidenced. ---
            let unused = t.never_selected(0);
            let frac = unused.len() as f64 / f64::from(t.n_experts);
            f.push(Finding {
                verdict: if frac > 0.05 {
                    Verdict::Marginal
                } else {
                    Verdict::Skip
                },
                topic: "expert pruning",
                evidence: format!(
                    "{} of {} experts were never selected in {} tokens ({:.0}% of streamed bytes)",
                    unused.len(),
                    t.n_experts,
                    t.n_tokens(),
                    frac * 100.0
                ),
                advice: if frac > 0.05 {
                    "Removing an expert removes its bytes entirely, which beats quantising it. \
                     But never-selected-in-this-trace is weaker evidence than a saliency \
                     measure: published methods score experts by gate-weighted output norms, \
                     not by frequency, precisely because a rarely-used expert can still be \
                     critical. Treat this as an upper bound on what is safe to drop, and only \
                     after the outlier-expert check above."
                        .into()
                } else {
                    "Nearly every expert is used by this workload, so there is little to remove \
                     without a real saliency analysis."
                        .into()
                },
            });

            // --- Segmented cache policy. Only answerable from a phased trace. ---
            if t.prefill_tokens == 0 {
                f.push(Finding {
                    verdict: Verdict::Unknown,
                    topic: "segmented cache",
                    evidence: format!(
                        "trace has no prefill/decode boundary ({} tokens, all decode)",
                        t.n_tokens()
                    ),
                    advice: "A segmented policy (SLRU) differs from LRU only across a prefill \
                             burst: prefill sweeps a wide set of experts once each, and a \
                             recency-only cache evicts the decode working set to make room for \
                             it. On an undifferentiated trace the two policies are the same \
                             measurement, so this cannot be answered. Capture a trace with the \
                             prompt included, or synthesise one with `synth --prefill N`."
                        .into(),
                });
            } else {
                let cap = cache_mib * 1024 * 1024;
                let cost = |bytes: u64| {
                    let bw = surface.bandwidth_at(layer.slice_bytes.max(4096), 16);
                    match bw {
                        Some(r) if r > 0.0 => bytes as f64 / r,
                        _ => 1.0,
                    }
                };
                // Replay decode only. Prefill's misses are unavoidable and
                // identical under every policy, so including them dilutes the
                // comparison with a constant.
                let decode_rate = |pol: pmx_cache::Policy| -> f64 {
                    let mut c = pmx_cache::ExpertCache::new(cap, pol);
                    let b = layer.bytes_per_expert;
                    // Key by (layer, expert). Expert 3 of layer 0 and expert 3 of
                    // layer 1 are different tensors; collapsing them onto one id
                    // shrinks the apparent working set and silently invents hits.
                    let replay = |c: &mut pmx_cache::ExpertCache, from: usize, to: usize| {
                        for tok in from..to {
                            for l in 0..t.n_layers {
                                for &e in t.selection(tok, l) {
                                    c.access((l, e), b, cost(b));
                                }
                            }
                        }
                    };
                    replay(&mut c, 0, t.prefill_tokens);
                    let before = c.stats();
                    replay(&mut c, t.prefill_tokens, t.n_tokens());
                    let a = c.stats();
                    let h = a.hits - before.hits;
                    let m = a.misses - before.misses;
                    if h + m == 0 {
                        0.0
                    } else {
                        h as f64 / (h + m) as f64
                    }
                };
                let lru = decode_rate(pmx_cache::Policy::Lru);
                let slru = decode_rate(pmx_cache::Policy::Slru);
                let gain = slru - lru;
                f.push(Finding {
                    verdict: if gain > 0.05 {
                        Verdict::Do
                    } else if gain > 0.01 {
                        Verdict::Marginal
                    } else {
                        Verdict::Skip
                    },
                    topic: "segmented cache",
                    evidence: format!(
                        "decode hit rate {:.1}% under LRU vs {:.1}% under SLRU \
                         ({} prefill / {} decode tokens)",
                        lru * 100.0,
                        slru * 100.0,
                        t.prefill_tokens,
                        t.decode_tokens()
                    ),
                    advice: if gain > 0.01 {
                        format!(
                            "Worth {:.1} points of decode hit rate here. Segmenting protects \
                             twice-used experts from a one-shot prefill sweep. Note the win is a \
                             one-off avoided reload of the working set, not an ongoing rate \
                             gain, so it shrinks as generation lengthens — it pays most on short \
                             replies to long prompts, and little on long generations.",
                            gain * 100.0
                        )
                    } else if gain < -0.05 {
                        format!(
                            "Do not segment here — it costs {:.1} points against plain LRU. \
                             Promotion needs a second access, and a second access needs the \
                             expert to survive probation, which is only {:.0}% of capacity. On \
                             this trace prefill promotes almost everything resident, protection \
                             fills, and the small probation left over turns over faster than a \
                             decode expert can be touched twice — so the protected segment \
                             freezes holding prefill's leftovers. `llama.cpp` #20757 proposes \
                             SLRU as its default; on this workload that default is a large \
                             regression, and this is worth telling them.",
                            -gain * 100.0,
                            (1.0 - pmx_cache::SLRU_PROTECTED_FRACTION) * 100.0
                        )
                    } else {
                        "Not worth the added state on this workload. Either the working set \
                         already fits alongside the prefill sweep, or decode runs long enough \
                         that the one-off reload LRU pays has amortised away. `llama.cpp` \
                         #20757 proposes SLRU as a default; on this trace that default would \
                         buy nothing measurable."
                            .to_string()
                    },
                });
            }

            f.push(Finding {
                verdict: classify(adj / chance.max(1e-9)),
                topic: "prefetch",
                evidence: format!(
                    "adjacent tokens reuse {:.0}% of experts against {:.0}% by chance",
                    adj * 100.0,
                    chance * 100.0
                ),
                advice: format!(
                    "Reuse is {:.2}x chance. Above ~1.5x there is real structure to predict; \
                     keep the budget near top-k, because every prediction is a real read and \
                     throughput peaks there then falls even as recall keeps rising. Near 1.0x \
                     a predictor cannot beat guessing the hottest experts.",
                    adj / chance.max(1e-9)
                ),
            });
        }
    }

    f.sort_by_key(|x| x.verdict.rank());
    for x in &f {
        println!("[{:^7}] {}", x.verdict.tag(), x.topic);
        println!("          {}", x.evidence);
        for line in wrap(&x.advice, 74) {
            println!("          {line}");
        }
        println!();
    }

    let count = |v: Verdict| f.iter().filter(|x| x.verdict == v).count();
    println!(
        "{} worth attempting, {} marginal, {} measurably not, {} undetermined.",
        count(Verdict::Do),
        count(Verdict::Marginal),
        count(Verdict::Skip),
        count(Verdict::Unknown)
    );
    println!(
        "\nThe SKIPs are the point. Each is an optimisation that sounds plausible, is\n\
         implemented in comparable tools, and does not pay on these inputs."
    );
    Ok(())
}

/// Sample one expert slice and measure its byte entropy.
fn sample_entropy(g: &Gguf, layer: &moe::MoeLayer) -> Option<(f64, u64)> {
    let name = layer.expert_tensors.first()?;
    let t = g.tensors.iter().find(|x| &x.name == name)?;
    if !can_dequantize(t.ggml_type.0) {
        return None;
    }
    let raw = g.read_tensor_slice(t, 0).ok()?;
    Some((entropy(&raw), raw.len() as u64))
}

/// Mean share of a token's experts that also appeared in the previous token.
fn adjacent_overlap(t: &Trace) -> f64 {
    let n = t.n_tokens();
    if n < 2 {
        return 0.0;
    }
    let mut shared = 0u64;
    let mut total = 0u64;
    for tok in 1..n {
        for l in 0..t.n_layers {
            let a = t.selection(tok - 1, l);
            let b = t.selection(tok, l);
            shared += b.iter().filter(|e| a.contains(e)).count() as u64;
            total += b.len() as u64;
        }
    }
    if total == 0 {
        0.0
    } else {
        shared as f64 / total as f64
    }
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.0} {}", U[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_borderline_signal_is_not_rounded_down_to_no() {
        // The bug this guards: a bare 1.5x threshold reported a genuine 1.49x
        // prefetch signal as a dead end. An advisor that overstates its
        // confidence at the boundary is worse than one that admits the boundary.
        assert_eq!(classify(1.49), Verdict::Marginal);
        assert_eq!(classify(1.20), Verdict::Marginal);
        assert_eq!(classify(1.50), Verdict::Do);
        assert_eq!(classify(3.00), Verdict::Do);
        assert_eq!(classify(1.05), Verdict::Skip);
        assert_eq!(classify(1.00), Verdict::Skip);
    }

    #[test]
    fn optimal_draft_depth_rises_with_acceptance() {
        // Emergent, not hardcoded: at low acceptance a deep draft wastes reads on
        // tokens that get rejected, so shallow wins; at high acceptance the union's
        // sublinearity pays and deep wins. If this inverts, the model is wrong.
        let t = pmx_trace::Trace::synthetic_cfg(&pmx_trace::SynthConfig {
            n_layers: 1,
            n_experts: 64,
            top_k: 8,
            tokens: 4000,
            clusters: 8,
            locality: 0.9,
            persistence: 0.85,
            layer_coupling: 0.0,
            prefill_tokens: 0,
            prefill_locality: 0.05,
            seed: 23,
        });
        let best_depth = |alpha: f64| -> usize {
            [2usize, 4, 8]
                .iter()
                .map(|&d| {
                    let union = t.mean_window_union(0, d);
                    let accepted: f64 = (0..d).map(|i| alpha.powi(i as i32 + 1)).sum::<f64>() + 1.0;
                    (d, union / accepted)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(d, _)| d)
                .unwrap()
        };
        let (lo, hi) = (best_depth(0.4), best_depth(0.95));
        assert!(
            hi >= lo,
            "optimal depth should not fall as acceptance rises: {lo} -> {hi}"
        );
        assert!(
            hi > lo,
            "expected a deeper optimum at high acceptance: {lo} -> {hi}"
        );
    }

    #[test]
    fn hazards_sort_above_everything_else() {
        // A quality hazard must not be buried under performance advice.
        let mut v = [
            Verdict::Skip,
            Verdict::Do,
            Verdict::Hazard,
            Verdict::Unknown,
            Verdict::Marginal,
        ];
        v.sort_by_key(|x| x.rank());
        assert_eq!(v[0], Verdict::Hazard);
        assert_eq!(v[1], Verdict::Do);
        assert_eq!(*v.last().unwrap(), Verdict::Skip);
    }

    #[test]
    fn entropy_of_uniform_bytes_is_eight_bits() {
        let all: Vec<u8> = (0..=255u8).collect();
        assert!((entropy(&all) - 8.0).abs() < 1e-9);
        // A constant stream carries no information.
        assert_eq!(entropy(&[7u8; 64]), 0.0);
        assert_eq!(entropy(&[]), 0.0);
    }

    #[test]
    fn entropy_separates_a_quantised_store_from_a_float_one() {
        // The measured distinction the "lossless coding" verdict rests on:
        // k-quant nibbles are near-uniform by construction (~7.8 bits/byte,
        // measured 2.8% redundant), whereas float weights have a redundant
        // exponent field (~28% redundant). Modelled here as skewed vs flat.
        let flat: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();
        let skewed: Vec<u8> = (0..4096u32)
            .map(|i| if i % 4 == 0 { (i % 256) as u8 } else { 0 })
            .collect();
        assert!(entropy(&flat) > 7.9, "{}", entropy(&flat));
        assert!(entropy(&skewed) < 3.0, "{}", entropy(&skewed));
    }

    #[test]
    fn adjacent_overlap_reflects_real_temporal_structure() {
        let none = pmx_trace::Trace::synthetic_cfg(&pmx_trace::SynthConfig {
            n_layers: 1,
            n_experts: 64,
            top_k: 8,
            tokens: 2000,
            clusters: 8,
            locality: 0.9,
            persistence: 0.0,
            layer_coupling: 0.0,
            prefill_tokens: 0,
            prefill_locality: 0.05,
            seed: 5,
        });
        let lots = pmx_trace::Trace::synthetic_cfg(&pmx_trace::SynthConfig {
            n_layers: 1,
            n_experts: 64,
            top_k: 8,
            tokens: 2000,
            clusters: 8,
            locality: 0.9,
            persistence: 0.95,
            layer_coupling: 0.0,
            prefill_tokens: 0,
            prefill_locality: 0.05,
            seed: 5,
        });
        let (a, b) = (adjacent_overlap(&none), adjacent_overlap(&lots));
        assert!(
            b > a * 1.5,
            "persistence should raise overlap: {a:.3} -> {b:.3}"
        );
    }

    #[test]
    fn wrapping_never_loses_or_splits_words() {
        let s = "one two three four five six seven eight nine ten eleven twelve";
        let out = wrap(s, 20);
        assert!(out.iter().all(|l| l.len() <= 20), "{out:?}");
        assert_eq!(out.join(" "), s);
    }
}
