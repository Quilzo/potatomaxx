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

/// Per-expert quality sensitivity, and the loss proxy applied to it.
#[derive(Debug, Clone)]
pub struct Sensitivity {
    /// Per-expert scale. Uniform 1.0 if nothing better is known.
    pub scale: Vec<f64>,
}

impl Sensitivity {
    /// Assume every expert is equally sensitive.
    pub fn uniform(n: usize) -> Self {
        Sensitivity {
            scale: vec![1.0; n],
        }
    }

    /// Proxy loss increase for storing expert `e` at `bits`.
    ///
    /// Quantisation error for a well-behaved quantiser falls as `4^-bits`; this
    /// is that shape, scaled per expert. It is a stand-in for measurement, and
    /// documented as such.
    pub fn delta_loss(&self, e: usize, bits: f64) -> f64 {
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
    /// Ceiling on proxy loss growth. Allocation stops crushing experts once
    /// expected loss would exceed `baseline_loss * loss_budget`.
    pub loss_budget: f64,
}

impl Default for PlanConfig {
    fn default() -> Self {
        PlanConfig {
            weights_per_expert: 0,
            resident_budget_bytes: 0,
            baseline_bits: 4.5,
            ladder: DEFAULT_LADDER.to_vec(),
            tier_cost: TierCost::default(),
            loss_budget: 1.10,
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
    let loss_ceiling = baseline_loss * cfg.loss_budget;

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
    fn allocation_respects_the_loss_budget() {
        let ca = skewed(32);
        let mut c = cfg(8);
        c.loss_budget = 1.02;
        let p = allocate(&ca, &Sensitivity::uniform(32), &c);
        assert!(
            p.loss_ratio() <= 1.02 + 1e-9,
            "proxy loss ratio {:.4} exceeded the 1.02 budget",
            p.loss_ratio()
        );
    }

    #[test]
    fn a_tighter_loss_budget_yields_less_speedup() {
        let ca = skewed(32);
        let mut tight = cfg(8);
        tight.loss_budget = 1.01;
        let mut loose = cfg(8);
        loose.loss_budget = 1.50;
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
    fn a_high_precision_checkpoint_keeps_the_no_change_option() {
        // An F16 checkpoint against a K-quant ladder: the top rung must be
        // "keep", so the loss ratio starts at 1.0 rather than exploding.
        let ca = skewed(32);
        let weights = 4 * 1024 * 1024u64;
        let c = PlanConfig {
            weights_per_expert: weights,
            resident_budget_bytes: 8 * bytes_at(weights, 16.0),
            baseline_bits: 16.0,
            loss_budget: 1.10,
            ..PlanConfig::default()
        };
        let p = allocate(&ca, &Sensitivity::uniform(32), &c);
        assert!(
            p.loss_ratio() <= 1.10 + 1e-9,
            "loss ratio {:.4} must respect the budget even when the ladder sits far below the checkpoint",
            p.loss_ratio()
        );
        assert!(
            p.speedup() >= 1.0,
            "allocation should never be slower than the baseline"
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
