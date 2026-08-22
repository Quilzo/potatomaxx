//! Deciding where each expert lives and how many bits it keeps.
//!
//! # The objective, and why it is not the usual one
//!
//! Mixed-precision quantisation normally allocates bits by *sensitivity*: how
//! much loss rises when a given tensor is crushed. That is the right objective
//! when the constraint is bytes **stored**.
//!
//! Here the constraint is bytes **moved**, per token, through a tiered store —
//! and the tiers are not equal. On the development machine, resident DRAM reads
//! run at ~30 GB/s while NVMe reads top out near 2.4 GB/s, so a byte fetched
//! from disk costs roughly 12x a byte fetched from RAM. An expert's contribution
//! to decode time is therefore proportional to
//!
//! ```text
//! frequency(e) x bytes(e, bits) x tier_cost(tier(e))
//! ```
//!
//! while its contribution to *quality* loss is proportional to
//! `frequency(e) x delta_loss(e, bits)` — an expert that is rarely selected
//! rarely affects the output. Minimising expected loss subject to a time budget
//! then says something the usual formulation does not: **spend fewer bits where
//! bytes are most expensive and least often needed.** Cold, disk-resident
//! experts should be quantised harder than hot, resident ones.
//!
//! # What is and is not implemented here
//!
//! This crate solves the *allocation* problem and reports the predicted byte
//! and time saving. It does **not** perform requantisation — turning a Q4_K
//! expert into a Q2_K one requires dequantise/requantise kernels and a quality
//! evaluation to confirm the predicted loss, and shipping the allocator without
//! that machinery would be shipping a number nobody had checked.
//!
//! The `delta_loss` term is likewise a **proxy**, not a measurement. The default
//! is the standard information-theoretic shape — error falls roughly as
//! `4^-bits` for a well-behaved quantiser — scaled per expert. Any real
//! deployment should replace it with measured per-expert sensitivity. The API
//! takes it as an input for exactly that reason.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pmx_trace::CoActivation;
use std::fmt;

/// Where an expert's weights live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Held in RAM for the whole session.
    Resident,
    /// Fetched from storage on demand.
    Storage,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Resident => write!(f, "resident"),
            Tier::Storage => write!(f, "storage"),
        }
    }
}

/// Relative cost of moving one byte from each tier.
#[derive(Debug, Clone, Copy)]
pub struct TierCost {
    /// Bytes per second achievable from RAM.
    pub resident_bps: f64,
    /// Bytes per second achievable from storage.
    pub storage_bps: f64,
}

impl Default for TierCost {
    fn default() -> Self {
        // Measured on the development machine: 30.0 GB/s DRAM (12 threads),
        // 2.41 GB/s NVMe random read at 2 MiB / QD8.
        TierCost {
            resident_bps: 30.0e9,
            storage_bps: 2.41e9,
        }
    }
}

impl TierCost {
    /// Seconds to move `bytes` from `tier`.
    pub fn seconds(&self, tier: Tier, bytes: u64) -> f64 {
        let bps = match tier {
            Tier::Resident => self.resident_bps,
            Tier::Storage => self.storage_bps,
        };
        if bps <= 0.0 {
            0.0
        } else {
            bytes as f64 / bps
        }
    }

    /// How much more a storage byte costs than a resident one.
    pub fn storage_penalty(&self) -> f64 {
        if self.storage_bps <= 0.0 {
            f64::INFINITY
        } else {
            self.resident_bps / self.storage_bps
        }
    }
}

/// A candidate precision an expert can be stored at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitOption {
    /// Nominal bits per weight, including quantisation scale overhead.
    pub bits: f64,
    /// A short label, e.g. `"Q4_K"`.
    pub label: &'static str,
}

/// The default ladder, ordered from most to least precise.
pub const DEFAULT_LADDER: &[BitOption] = &[
    BitOption {
        bits: 6.5625,
        label: "Q6_K",
    },
    BitOption {
        bits: 5.5,
        label: "Q5_K",
    },
    BitOption {
        bits: 4.5,
        label: "Q4_K",
    },
    BitOption {
        bits: 3.4375,
        label: "Q3_K",
    },
    BitOption {
        bits: 2.625,
        label: "Q2_K",
    },
];

/// How much quality each expert loses at each precision.
///
/// Two sources are supported, and which one is in use matters a great deal:
///
/// * [`Sensitivity::measured`] carries real per-expert, per-precision error —
///   normally the round-trip RMSE of actually quantising that expert's weights.
///   This is what `potatomaxx build-store` uses.
/// * [`Sensitivity::uniform`] falls back to an analytic proxy, `4^-bits`.
///
/// The proxy is a poor absolute model and must not be trusted to set a loss
/// budget. Against an F16 baseline it reports an enormous loss for any
/// quantisation at all, so a budget of 1.10 refuses every demotion and the
/// allocator silently does nothing — reporting a 1.00x speedup and a perfect
/// loss ratio while having changed not one bit. That is exactly what happened
/// before measurement was wired in, which is why the distinction is surfaced
/// here rather than buried in a default.
#[derive(Debug, Clone)]
pub struct Sensitivity {
    /// Per-expert scale, used only by the analytic proxy.
    pub scale: Vec<f64>,
    /// `measured[e]` maps bits to observed error for expert `e`.
    measured: Vec<Vec<(f64, f64)>>,
}

impl Sensitivity {
    /// Fall back to the analytic proxy, assuming equal sensitivity.
    pub fn uniform(n: usize) -> Self {
        Sensitivity {
            scale: vec![1.0; n],
            measured: Vec::new(),
        }
    }

    /// Use measured error. `per_expert[e]` is a list of `(bits, error)` pairs.
    ///
    /// Non-finite entries are dropped. Admitting one would make every comparison
    /// against it false, so the allocator would quietly refuse every demotion and
    /// still report success — the failure is invisible unless you look for it.
    pub fn measured(per_expert: Vec<Vec<(f64, f64)>>) -> Self {
        let n = per_expert.len();
        let measured = per_expert
            .into_iter()
            .map(|rows| {
                rows.into_iter()
                    .filter(|(b, e)| b.is_finite() && e.is_finite())
                    .collect()
            })
            .collect();
        Sensitivity {
            scale: vec![1.0; n],
            measured,
        }
    }

    /// Whether this carries real measurements rather than the proxy.
    pub fn is_measured(&self) -> bool {
        !self.measured.is_empty()
    }

    /// Loss increase for storing expert `e` at `bits`.
    ///
    /// Measured values snap to the nearest recorded bit width; the proxy is used
    /// only where no measurement exists.
    pub fn delta_loss(&self, e: usize, bits: f64) -> f64 {
        if let Some(rows) = self.measured.get(e) {
            if !rows.is_empty() {
                let mut best = rows[0];
                let mut bd = (rows[0].0 - bits).abs();
                for r in rows.iter().skip(1) {
                    let d = (r.0 - bits).abs();
                    if d < bd {
                        bd = d;
                        best = *r;
                    }
                }
                return best.1;
            }
        }
        let s = self.scale.get(e).copied().unwrap_or(1.0);
        s * 4f64.powf(-bits)
    }
}

/// One expert's allocation.
#[derive(Debug, Clone, Copy)]
pub struct ExpertAlloc {
    /// Expert index within its layer.
    pub expert: u32,
    /// Selections observed in the trace.
    pub freq: u64,
    /// Assigned tier.
    pub tier: Tier,
    /// Chosen precision.
    pub bits: f64,
    /// Label of the chosen precision.
    pub label: &'static str,
    /// Bytes at the chosen precision.
    pub bytes: u64,
}

/// The allocation for one layer, plus what it is predicted to buy.
#[derive(Debug, Clone)]
pub struct AllocPlan {
    /// Per-expert allocations, expert-indexed.
    pub experts: Vec<ExpertAlloc>,
    /// Bytes held resident.
    pub resident_bytes: u64,
    /// Predicted seconds per token of weight movement, before allocation.
    pub baseline_seconds: f64,
    /// Predicted seconds per token after allocation.
    pub planned_seconds: f64,
    /// Proxy expected loss before allocation.
    pub baseline_loss: f64,
    /// Proxy expected loss after allocation.
    pub planned_loss: f64,
}

impl AllocPlan {
    /// Predicted speedup in weight movement.
    pub fn speedup(&self) -> f64 {
        if self.planned_seconds <= 0.0 {
            1.0
        } else {
            self.baseline_seconds / self.planned_seconds
        }
    }

    /// Proxy loss increase, as a multiple of the baseline. 1.0 means unchanged.
    pub fn loss_ratio(&self) -> f64 {
        if self.baseline_loss <= 0.0 {
            1.0
        } else {
            self.planned_loss / self.baseline_loss
        }
    }
}

/// How much quality loss an allocation may spend.
///
/// A multiplier on the baseline's loss is the obvious formulation and it is
/// degenerate: once sensitivity is *measured*, the baseline is the reference and
/// its error is zero, so any multiple of it is still zero and no demotion is ever
/// permitted. [`ErrorBudget::UniformAt`] avoids that and is also how a
/// practitioner actually reasons — "no worse than storing everything at 4 bits".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorBudget {
    /// Allow at most this frequency-weighted expected error, in the units of the
    /// sensitivity measurement.
    Absolute(f64),
    /// Allow at most the error a uniform allocation at `bits` would incur.
    UniformAt(f64),
    /// Allow at most `factor` times the baseline's error.
    ///
    /// Only meaningful when the baseline is itself lossy relative to the
    /// sensitivity reference.
    RelativeToBaseline(f64),
}

impl ErrorBudget {
    /// Resolve to an absolute ceiling.
    fn ceiling(
        self,
        n: usize,
        rate: &dyn Fn(usize) -> f64,
        sens: &Sensitivity,
        baseline_bits: f64,
    ) -> f64 {
        let weighted =
            |bits: f64| -> f64 { (0..n).map(|e| rate(e) * sens.delta_loss(e, bits)).sum() };
        match self {
            ErrorBudget::Absolute(v) => v,
            ErrorBudget::UniformAt(bits) => weighted(bits),
            ErrorBudget::RelativeToBaseline(f) => weighted(baseline_bits) * f,
        }
    }
}

/// Inputs to allocation.
#[derive(Debug, Clone)]
pub struct PlanConfig {
    /// Weights per expert (elements, not bytes).
    pub weights_per_expert: u64,
    /// Bytes of RAM available for routed experts in this layer.
    pub resident_budget_bytes: u64,
    /// Precision the checkpoint currently uses, for the baseline comparison.
    pub baseline_bits: f64,
    /// Precisions available.
    pub ladder: Vec<BitOption>,
    /// Tier bandwidths.
    pub tier_cost: TierCost,
    /// How much quality loss the allocation may spend.
    pub error_budget: ErrorBudget,
}

impl Default for PlanConfig {
    fn default() -> Self {
        PlanConfig {
            weights_per_expert: 0,
            resident_budget_bytes: 0,
            baseline_bits: 4.5,
            ladder: DEFAULT_LADDER.to_vec(),
            tier_cost: TierCost::default(),
            error_budget: ErrorBudget::UniformAt(4.5),
        }
    }
}

fn bytes_at(weights: u64, bits: f64) -> u64 {
    ((weights as f64 * bits) / 8.0).ceil() as u64
}

/// Assign tiers by observed frequency, then allocate bits.
///
/// Residency is greedy by frequency: the most-selected experts fill the RAM
/// budget, since they are the ones whose repeated fetches dominate. Bit
/// allocation is then a Lagrangian sweep — repeatedly demote whichever expert
/// offers the best time-saved per unit of proxy loss added, until either the
/// ladder bottoms out or the loss budget is reached.
pub fn allocate(ca: &CoActivation, sens: &Sensitivity, cfg: &PlanConfig) -> AllocPlan {
    let n = ca.n_experts as usize;
    let mut ladder = cfg.ladder.clone();
    ladder.sort_by(|a, b| {
        b.bits
            .partial_cmp(&a.bits)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // "Leave it alone" must always be the top rung. Otherwise, for a checkpoint
    // stored above the ladder's ceiling (an F16 model against a K-quant ladder),
    // every expert starts already degraded and the loss budget polices
    // demotions from a baseline nobody chose. Keeping the source precision on
    // the ladder makes the no-change option reachable and the budget meaningful.
    if ladder
        .first()
        .map_or(true, |top| cfg.baseline_bits > top.bits)
    {
        ladder.insert(
            0,
            BitOption {
                bits: cfg.baseline_bits,
                label: "keep",
            },
        );
    }

    // Residency: highest frequency first, until the budget is exhausted. Sizing
    // uses the *baseline* precision, since residency is decided before bits.
    let base_bytes = bytes_at(cfg.weights_per_expert, cfg.baseline_bits);
    let mut by_freq: Vec<u32> = (0..n as u32).collect();
    by_freq.sort_by_key(|&e| std::cmp::Reverse(ca.freq[e as usize]));
    let mut tier = vec![Tier::Storage; n];
    let mut resident_bytes = 0u64;
    for &e in &by_freq {
        if base_bytes > 0 && resident_bytes + base_bytes <= cfg.resident_budget_bytes {
            tier[e as usize] = Tier::Resident;
            resident_bytes += base_bytes;
        }
    }

    // Start everyone at the top of the ladder, then demote greedily.
    let mut level = vec![0usize; n];
    let total_tokens = ca.tokens.max(1) as f64;

    let seconds_of = |e: usize, lvl: usize, tier: Tier| -> f64 {
        let b = bytes_at(cfg.weights_per_expert, ladder[lvl].bits);
        let rate = ca.freq[e] as f64 / total_tokens;
        rate * cfg.tier_cost.seconds(tier, b)
    };
    let loss_of = |e: usize, lvl: usize| -> f64 {
        let rate = ca.freq[e] as f64 / total_tokens;
        rate * sens.delta_loss(e, ladder[lvl].bits)
    };

    let baseline_seconds: f64 = (0..n)
        .map(|e| {
            let rate = ca.freq[e] as f64 / total_tokens;
            rate * cfg.tier_cost.seconds(tier[e], base_bytes)
        })
        .sum();
    let baseline_loss: f64 = (0..n)
        .map(|e| {
            let rate = ca.freq[e] as f64 / total_tokens;
            rate * sens.delta_loss(e, cfg.baseline_bits)
        })
        .sum();

    let mut cur_loss: f64 = (0..n).map(|e| loss_of(e, 0)).sum();
    let rate_of = |e: usize| ca.freq[e] as f64 / total_tokens;
    let loss_ceiling = cfg
        .error_budget
        .ceiling(n, &rate_of, sens, cfg.baseline_bits);

    loop {
        let mut best: Option<(f64, usize)> = None;
        for e in 0..n {
            let lvl = level[e];
            if lvl + 1 >= ladder.len() {
                continue;
            }
            let time_saved = seconds_of(e, lvl, tier[e]) - seconds_of(e, lvl + 1, tier[e]);
            let loss_added = loss_of(e, lvl + 1) - loss_of(e, lvl);
            if loss_added <= 0.0 {
                // Free improvement; take it immediately.
                best = Some((f64::INFINITY, e));
                break;
            }
            if cur_loss + loss_added > loss_ceiling {
                continue;
            }
            let ratio = time_saved / loss_added;
            let better = match best {
                None => true,
                Some((br, _)) => ratio > br,
            };
            if ratio > 0.0 && better {
                best = Some((ratio, e));
            }
        }
        match best {
            Some((_, e)) => {
                let lvl = level[e];
                cur_loss += loss_of(e, lvl + 1) - loss_of(e, lvl);
                level[e] = lvl + 1;
            }
            None => break,
        }
    }

    let experts: Vec<ExpertAlloc> = (0..n)
        .map(|e| {
            let opt = ladder[level[e]];
            ExpertAlloc {
                expert: e as u32,
                freq: ca.freq[e],
                tier: tier[e],
                bits: opt.bits,
                label: opt.label,
                bytes: bytes_at(cfg.weights_per_expert, opt.bits),
            }
        })
        .collect();

    // Recompute residency in the allocated precisions, for reporting.
    let resident_bytes = experts
        .iter()
        .filter(|a| a.tier == Tier::Resident)
        .map(|a| a.bytes)
        .sum();
    let planned_seconds: f64 = (0..n).map(|e| seconds_of(e, level[e], tier[e])).sum();
    let planned_loss: f64 = (0..n).map(|e| loss_of(e, level[e])).sum();

    AllocPlan {
        experts,
        resident_bytes,
        baseline_seconds,
        planned_seconds,
        baseline_loss,
        planned_loss,
    }
}

/// Share of selections served from RAM under this plan.
pub fn hit_rate(plan: &AllocPlan) -> f64 {
    let total: u64 = plan.experts.iter().map(|a| a.freq).sum();
    if total == 0 {
        return 0.0;
    }
    let hits: u64 = plan
        .experts
        .iter()
        .filter(|a| a.tier == Tier::Resident)
        .map(|a| a.freq)
        .sum();
    hits as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmx_trace::Trace;

    fn skewed(n: u32) -> CoActivation {
        // Clustered routing concentrates frequency on a subset of experts.
        let t = Trace::synthetic(1, n, 4, 4000, 4, 1.0, 17);
        CoActivation::from_trace(&t, 0)
    }

    /// Frequency-weighted expected loss if every expert were stored at `bits`.
    fn uniform_loss(ca: &CoActivation, sens: &Sensitivity, bits: f64) -> f64 {
        let total = ca.tokens.max(1) as f64;
        (0..ca.n_experts as usize)
            .map(|e| (ca.freq[e] as f64 / total) * sens.delta_loss(e, bits))
            .sum()
    }

    fn cfg(n_resident_experts: u64) -> PlanConfig {
        let weights = 4 * 1024 * 1024u64;
        PlanConfig {
            weights_per_expert: weights,
            resident_budget_bytes: n_resident_experts * bytes_at(weights, 4.5),
            baseline_bits: 4.5,
            ..PlanConfig::default()
        }
    }

    #[test]
    fn non_finite_measurements_are_dropped() {
        let sens = Sensitivity::measured(vec![vec![
            (16.0, 0.0),
            (4.5, f64::NAN),
            (2.5, f64::INFINITY),
        ]]);
        // Only the finite row survives, so every query resolves to it.
        assert!(sens.delta_loss(0, 4.5).is_finite());
        assert_eq!(sens.delta_loss(0, 4.5), 0.0);
    }

    #[test]
    fn measured_sensitivity_overrides_the_proxy() {
        // Two experts with deliberately opposite error profiles: expert 0 is
        // robust at low precision, expert 1 is not. The allocator must follow the
        // measurement, not the uniform proxy.
        let measured = vec![
            vec![(8.25, 0.0), (4.5, 0.001), (2.5, 0.002)],
            vec![(8.25, 0.0), (4.5, 0.500), (2.5, 2.000)],
        ];
        let sens = Sensitivity::measured(measured);
        assert!(sens.is_measured());
        assert!(sens.delta_loss(1, 2.5) > sens.delta_loss(0, 2.5) * 100.0);
        // Nearest-bits lookup, not the proxy.
        assert!((sens.delta_loss(0, 4.4) - 0.001).abs() < 1e-12);
        // Unmeasured index falls back to the proxy rather than panicking.
        assert!(sens.delta_loss(9, 4.0) > 0.0);
        assert!(!Sensitivity::uniform(2).is_measured());
    }

    #[test]
    fn residency_goes_to_the_hottest_experts() {
        let ca = skewed(32);
        let p = allocate(&ca, &Sensitivity::uniform(32), &cfg(8));
        let resident_min = p
            .experts
            .iter()
            .filter(|a| a.tier == Tier::Resident)
            .map(|a| a.freq)
            .min()
            .unwrap();
        let storage_max = p
            .experts
            .iter()
            .filter(|a| a.tier == Tier::Storage)
            .map(|a| a.freq)
            .max()
            .unwrap();
        assert!(
            resident_min >= storage_max,
            "coldest resident expert ({resident_min}) should not be colder than the hottest evicted one ({storage_max})"
        );
    }

    #[test]
    fn hit_rate_exceeds_the_uniform_share_when_routing_is_skewed() {
        let ca = skewed(64);
        let p = allocate(&ca, &Sensitivity::uniform(64), &cfg(16));
        // 16 of 64 experts resident would be 25% under uniform routing.
        assert!(
            hit_rate(&p) > 0.25,
            "skewed routing should beat the uniform share, got {:.3}",
            hit_rate(&p)
        );
    }

    #[test]
    fn cold_experts_are_quantised_at_least_as_hard_as_hot_ones() {
        let ca = skewed(48);
        let p = allocate(&ca, &Sensitivity::uniform(48), &cfg(12));
        let hot_mean: f64 = {
            let v: Vec<f64> = p
                .experts
                .iter()
                .filter(|a| a.tier == Tier::Resident)
                .map(|a| a.bits)
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let cold_mean: f64 = {
            let v: Vec<f64> = p
                .experts
                .iter()
                .filter(|a| a.tier == Tier::Storage)
                .map(|a| a.bits)
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(
            cold_mean <= hot_mean + 1e-9,
            "storage-tier experts ({cold_mean:.2} bits) should not keep more bits than resident ones ({hot_mean:.2})"
        );
    }

    #[test]
    fn allocation_respects_a_tight_precision_ceiling() {
        let ca = skewed(32);
        let mut c = cfg(8);
        c.error_budget = ErrorBudget::UniformAt(6.0);
        let sens = Sensitivity::uniform(32);
        let p = allocate(&ca, &sens, &c);
        let ceiling = uniform_loss(&ca, &sens, 6.0);
        assert!(
            p.planned_loss <= ceiling * 1.000_001,
            "expected loss {} exceeded the uniform-6-bit ceiling {ceiling}",
            p.planned_loss
        );
    }

    #[test]
    fn a_tighter_precision_ceiling_yields_less_speedup() {
        let ca = skewed(32);
        let mut tight = cfg(8);
        tight.error_budget = ErrorBudget::UniformAt(6.5);
        let mut loose = cfg(8);
        loose.error_budget = ErrorBudget::UniformAt(2.5);
        let a = allocate(&ca, &Sensitivity::uniform(32), &tight);
        let b = allocate(&ca, &Sensitivity::uniform(32), &loose);
        assert!(
            b.speedup() >= a.speedup(),
            "loosening the loss budget should not reduce speedup: {:.3} vs {:.3}",
            b.speedup(),
            a.speedup()
        );
    }

    #[test]
    fn allocation_stays_within_a_uniform_precision_ceiling() {
        // The budget's promise: the result is no worse than storing everything at
        // the named bit width. It says nothing about being close to the source,
        // which is the caller's choice to make.
        let ca = skewed(32);
        let sens = Sensitivity::uniform(32);
        let weights = 4 * 1024 * 1024u64;
        let c = PlanConfig {
            weights_per_expert: weights,
            resident_budget_bytes: 8 * bytes_at(weights, 16.0),
            baseline_bits: 16.0,
            error_budget: ErrorBudget::UniformAt(4.5),
            ..PlanConfig::default()
        };
        let p = allocate(&ca, &sens, &c);
        let ceiling = uniform_loss(&ca, &sens, 4.5);
        assert!(
            p.planned_loss <= ceiling * 1.000_001,
            "expected loss {} exceeded the uniform-4.5-bit ceiling {ceiling}",
            p.planned_loss
        );
        assert!(
            p.speedup() >= 1.0,
            "allocation should never be slower than the baseline"
        );
    }

    #[test]
    fn a_budget_relative_to_the_baseline_is_degenerate_under_measurement() {
        // Documents why ErrorBudget::UniformAt exists. Once sensitivity is
        // measured, the baseline *is* the reference and its error is zero, so any
        // multiple of it is still zero: RelativeToBaseline permits nothing and the
        // allocator silently does nothing at all. Before this was understood, the
        // tool reported a 1.00x speedup and a perfect loss ratio while having
        // changed not one bit.
        let ca = skewed(32);
        let rows: Vec<Vec<(f64, f64)>> = (0..32)
            .map(|_| vec![(16.0, 0.0), (8.25, 0.001), (4.5, 0.002), (2.5, 0.004)])
            .collect();
        let sens = Sensitivity::measured(rows);
        let weights = 4 * 1024 * 1024u64;
        let base = PlanConfig {
            weights_per_expert: weights,
            resident_budget_bytes: 8 * bytes_at(weights, 16.0),
            baseline_bits: 16.0,
            ..PlanConfig::default()
        };

        let degenerate = allocate(
            &ca,
            &sens,
            &PlanConfig {
                error_budget: ErrorBudget::RelativeToBaseline(1.10),
                ..base.clone()
            },
        );
        assert!(
            (degenerate.speedup() - 1.0).abs() < 1e-9,
            "a baseline-relative budget should permit nothing here, got {:.3}x",
            degenerate.speedup()
        );

        let usable = allocate(
            &ca,
            &sens,
            &PlanConfig {
                error_budget: ErrorBudget::UniformAt(4.5),
                ..base.clone()
            },
        );
        assert!(
            usable.speedup() > 1.5,
            "a uniform-precision budget should unlock real demotion, got {:.3}x",
            usable.speedup()
        );
    }

    #[test]
    fn an_absolute_budget_is_honoured_exactly() {
        let ca = skewed(24);
        let rows: Vec<Vec<(f64, f64)>> = (0..24)
            .map(|_| vec![(16.0, 0.0), (8.25, 0.01), (4.5, 0.02), (2.5, 0.05)])
            .collect();
        let sens = Sensitivity::measured(rows);
        let weights = 1024 * 1024u64;
        let p = allocate(
            &ca,
            &sens,
            &PlanConfig {
                weights_per_expert: weights,
                resident_budget_bytes: 4 * bytes_at(weights, 16.0),
                baseline_bits: 16.0,
                error_budget: ErrorBudget::Absolute(0.015),
                ..PlanConfig::default()
            },
        );
        assert!(
            p.planned_loss <= 0.015 + 1e-12,
            "expected loss {} exceeded the absolute budget 0.015",
            p.planned_loss
        );
    }

    #[test]
    fn storage_penalty_reflects_measured_tiers() {
        let tc = TierCost::default();
        assert!((tc.storage_penalty() - 30.0e9 / 2.41e9).abs() < 1e-6);
        assert!(tc.storage_penalty() > 12.0 && tc.storage_penalty() < 12.6);
    }

    #[test]
    fn zero_budget_leaves_everything_on_storage() {
        let ca = skewed(16);
        let p = allocate(&ca, &Sensitivity::uniform(16), &cfg(0));
        assert!(p.experts.iter().all(|a| a.tier == Tier::Storage));
        assert_eq!(hit_rate(&p), 0.0);
    }
}
