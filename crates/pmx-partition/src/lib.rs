// SPDX-License-Identifier: GPL-2.0-or-later
//! Choosing an expert order, and honestly costing the choice.
//!
//! # What layout can and cannot do
//!
//! A GGUF MoE layer stacks its experts into a handful of tensors — typically
//! three (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) — with the expert
//! index as the last axis. Fetching the top-k experts for one token therefore
//! means reading k slices out of each of those tensors: `k * n_tensors`
//! scattered requests.
//!
//! Permuting the expert axis so that experts which fire together sit next to
//! each other turns those scattered slices into a few contiguous **runs**. The
//! same bytes are read, in fewer and larger requests. Nothing is over-read, and
//! because a permutation is only a relabelling (see `pmx_gguf::write`), the
//! model computes exactly the same function.
//!
//! It is worth being precise about the size of that win, because it is easy to
//! oversell. Measured on the development machine's NVMe:
//!
//! ```text
//! slice size   scattered (24 req)   coalesced (6 req)   gain
//!    16 KiB               0.29 ms             0.48 ms   none — already fine
//!    64 KiB               1.16 ms             0.73 ms   1.58x
//!   128 KiB               2.31 ms             1.46 ms   1.58x
//!   256 KiB               3.11 ms             2.85 ms   1.09x
//! ```
//!
//! So: a real but modest win, concentrated in a band of slice sizes. Meanwhile
//! the same device goes from 0.26 GB/s at queue depth 1 to 2.15 GB/s at queue
//! depth 8 — roughly **8x** — and no file layout can influence that. Queue
//! depth is won by a runtime that knows which experts it needs early enough to
//! have many reads outstanding; it is deliberately out of scope here.
//!
//! This crate therefore does not assume a win. It costs the candidate order
//! against a *measured* [`Surface`] and reports what the gain actually is for
//! this model, this trace and this device — including when the answer is
//! "nothing useful, keep the checkpoint order".

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pmx_probe::Surface;
use pmx_trace::{CoActivation, Trace};

/// How reads are costed.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Bytes in one expert's slice of one tensor.
    pub slice_bytes: u64,
    /// How many stacked tensors carry the expert axis for this layer.
    pub tensors_per_expert: u32,
    /// Runs separated by at most this many unused slices are merged into one
    /// request, trading a little over-read for one fewer request. Zero disables.
    pub merge_gap_slices: u32,
    /// Read requests the *runtime* is assumed to keep in flight.
    ///
    /// This is a property of the engine consuming the file, not of the layout,
    /// so it is a parameter rather than something derived from how many requests
    /// a single token happens to need. Deriving it from request count would let
    /// the optimiser "win" by scattering reads to reach a deeper-queue cell of
    /// the measured surface, which is an artifact of the model rather than a
    /// real speedup.
    pub queue_depth: usize,
    /// Measured device bandwidth surface.
    pub surface: Surface,
}

impl CostModel {
    /// A cost model with no measured surface, for structural tests only.
    ///
    /// Costing falls back to counting requests, which is monotone in the real
    /// objective but not calibrated to any device.
    pub fn uncalibrated(slice_bytes: u64, tensors_per_expert: u32) -> Self {
        CostModel {
            slice_bytes,
            tensors_per_expert,
            merge_gap_slices: 0,
            queue_depth: 8,
            surface: Surface::default(),
        }
    }
}

/// The predicted cost of fetching experts under a given order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// Predicted seconds of read time per token, summed over the layer's tensors.
    pub seconds_per_token: f64,
    /// Mean bytes read per token, including any over-read from merged runs.
    pub bytes_per_token: f64,
    /// Mean number of read requests issued per token.
    pub requests_per_token: f64,
}

impl Cost {
    /// Speedup of `self` relative to `baseline`. Above 1.0 means `self` is faster.
    pub fn speedup_over(&self, baseline: &Cost) -> f64 {
        if self.seconds_per_token <= 0.0 {
            return 1.0;
        }
        baseline.seconds_per_token / self.seconds_per_token
    }
}

/// An expert ordering for one layer.
///
/// `slot_of[e]` is the physical slot expert `e` occupies. `expert_at[s]` is the
/// inverse. `expert_at` is exactly the permutation `pmx_gguf` consumes: new slot
/// `s` receives old expert `expert_at[s]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    slot_of: Vec<u32>,
    expert_at: Vec<u32>,
}

impl Layout {
    /// The order a checkpoint already has.
    pub fn identity(n: u32) -> Self {
        Layout {
            slot_of: (0..n).collect(),
            expert_at: (0..n).collect(),
        }
    }

    /// Build from a permutation where `expert_at[s] == e`.
    ///
    /// Returns `None` unless the input is a true permutation of `0..n`.
    pub fn from_expert_at(expert_at: Vec<u32>) -> Option<Self> {
        let n = expert_at.len();
        let mut slot_of = vec![u32::MAX; n];
        for (slot, &e) in expert_at.iter().enumerate() {
            let e = e as usize;
            if e >= n || slot_of[e] != u32::MAX {
                return None;
            }
            slot_of[e] = slot as u32;
        }
        Some(Layout { slot_of, expert_at })
    }

    /// Number of experts.
    pub fn len(&self) -> usize {
        self.expert_at.len()
    }

    /// Whether the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.expert_at.is_empty()
    }

    /// The permutation to hand to the repacker: new slot `s` takes old expert
    /// `expert_at()[s]`.
    pub fn expert_at(&self) -> &[u32] {
        &self.expert_at
    }

    /// Slot occupied by each expert.
    pub fn slot_of(&self) -> &[u32] {
        &self.slot_of
    }

    fn swap_slots(&mut self, a: usize, b: usize) {
        let (ea, eb) = (self.expert_at[a], self.expert_at[b]);
        self.expert_at.swap(a, b);
        self.slot_of[ea as usize] = b as u32;
        self.slot_of[eb as usize] = a as u32;
    }
}

/// Hyperedges: which experts each token selected, flattened.
#[derive(Debug, Clone)]
pub struct Edges {
    flat: Vec<u32>,
    k: usize,
    /// For each expert, the indices of edges that contain it.
    by_expert: Vec<Vec<u32>>,
}

impl Edges {
    /// Collect one layer's selections from a trace.
    pub fn from_trace(trace: &Trace, layer: u32) -> Self {
        let k = trace.top_k as usize;
        let mut flat = Vec::with_capacity(trace.n_tokens() * k);
        for e in trace.layer_edges(layer) {
            flat.extend_from_slice(e);
        }
        let mut by_expert = vec![Vec::new(); trace.n_experts as usize];
        for (i, chunk) in flat.chunks_exact(k).enumerate() {
            for &e in chunk {
                let v = &mut by_expert[e as usize];
                if v.last() != Some(&(i as u32)) {
                    v.push(i as u32);
                }
            }
        }
        Edges { flat, k, by_expert }
    }

    /// Number of edges (tokens).
    pub fn len(&self) -> usize {
        self.flat.len().checked_div(self.k).unwrap_or(0)
    }

    /// Whether there are no edges.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn edge(&self, i: usize) -> &[u32] {
        &self.flat[i * self.k..(i + 1) * self.k]
    }
}

/// Cost of a single edge under a layout. Returns `(bytes, requests)`.
fn edge_shape(edge: &[u32], layout: &Layout, cm: &CostModel, scratch: &mut Vec<u32>) -> (u64, u64) {
    scratch.clear();
    for &e in edge {
        scratch.push(layout.slot_of[e as usize]);
    }
    scratch.sort_unstable();
    scratch.dedup();

    let gap = u64::from(cm.merge_gap_slices);
    let mut runs: u64 = 0;
    let mut span_slices: u64 = 0;
    let mut i = 0;
    while i < scratch.len() {
        let start = u64::from(scratch[i]);
        let mut end = start;
        i += 1;
        // Extend the run while the next needed slot is within the merge gap.
        while i < scratch.len() {
            let next = u64::from(scratch[i]);
            if next <= end + 1 + gap {
                end = next;
                i += 1;
            } else {
                break;
            }
        }
        runs += 1;
        span_slices += end - start + 1;
    }
    let tpe = u64::from(cm.tensors_per_expert).max(1);
    (span_slices * cm.slice_bytes * tpe, runs * tpe)
}

/// Predicted cost of one layer's fetches under `layout`.
pub fn evaluate(edges: &Edges, layout: &Layout, cm: &CostModel) -> Cost {
    let n = edges.len();
    if n == 0 {
        return Cost {
            seconds_per_token: 0.0,
            bytes_per_token: 0.0,
            requests_per_token: 0.0,
        };
    }
    let mut scratch = Vec::with_capacity(edges.k);
    let mut secs = 0.0f64;
    let mut bytes_acc = 0u64;
    let mut req_acc = 0u64;
    for i in 0..n {
        let (bytes, reqs) = edge_shape(edges.edge(i), layout, cm, &mut scratch);
        bytes_acc += bytes;
        req_acc += reqs;
        secs += seconds_for(bytes, reqs, cm);
    }
    Cost {
        seconds_per_token: secs / n as f64,
        bytes_per_token: bytes_acc as f64 / n as f64,
        requests_per_token: req_acc as f64 / n as f64,
    }
}

/// Predicted seconds to move `bytes` in `requests` requests.
///
/// Bandwidth is read at the *mean request size* and the runtime's assumed queue
/// depth. At fixed queue depth the measured surface rises monotonically with
/// request size, so coalescing scattered slices into fewer, larger reads is
/// always at least as fast — which is the property the optimiser exploits.
fn seconds_for(bytes: u64, requests: u64, cm: &CostModel) -> f64 {
    if requests == 0 || bytes == 0 {
        return 0.0;
    }
    let mean_req = bytes / requests;
    match cm.surface.bandwidth_at(mean_req, cm.queue_depth) {
        Some(bw) if bw > 0.0 => bytes as f64 / bw,
        // With no measured surface, fall back to request count. Monotone in the
        // real objective but not calibrated — never report this as a time.
        _ => requests as f64,
    }
}

/// Search settings.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Maximum refinement passes.
    pub max_passes: usize,
    /// Candidate swap partners considered per expert per pass.
    pub candidates: usize,
    /// Stop when a pass improves the objective by less than this fraction.
    pub min_improvement: f64,
    /// Cap on affected edges scored when costing one candidate swap.
    ///
    /// Below this the delta is exact. Above it a deterministic stride sample is
    /// scored and scaled, which keeps a pass linear in expert count rather than
    /// in trace length. Sampling makes candidate scores approximate, so each
    /// pass is confirmed by a full evaluation and rolled back if it regressed.
    pub max_delta_edges: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        OptimizeConfig {
            max_passes: 6,
            candidates: 12,
            min_improvement: 0.001,
            max_delta_edges: 256,
        }
    }
}

/// What the optimiser found.
#[derive(Debug, Clone)]
pub struct OptimizeReport {
    /// The chosen order.
    pub layout: Layout,
    /// Cost of the checkpoint's existing order.
    pub baseline: Cost,
    /// Cost after seeding, before local refinement.
    pub seeded: Cost,
    /// Cost of the chosen order.
    pub optimized: Cost,
    /// Refinement passes actually run.
    pub passes: usize,
    /// Swaps accepted.
    pub swaps: usize,
}

impl OptimizeReport {
    /// Predicted speedup over the checkpoint's order.
    pub fn speedup(&self) -> f64 {
        self.optimized.speedup_over(&self.baseline)
    }

    /// Whether the gain clears `min_speedup` and is worth rewriting a file for.
    ///
    /// Below the threshold the honest recommendation is to leave the checkpoint
    /// alone: a few percent is inside the noise of the bandwidth measurement
    /// itself. Some devices offer nothing at all — throttled cloud storage
    /// measures nearly flat across request sizes, and on such a device no
    /// layout can help. Saying so is the point, not a failure.
    pub fn worth_repacking(&self, min_speedup: f64) -> bool {
        self.speedup() >= min_speedup
    }
}

/// Seed an order by chaining experts together in descending co-activation.
///
/// Start from the most-co-activated pair, then repeatedly append whichever
/// unplaced expert has the strongest affinity to the tail of the chain. This is
/// a greedy seriation: cheap, deterministic, and good enough that refinement
/// has little left to do.
pub fn seed_by_coactivation(ca: &CoActivation) -> Layout {
    let n = ca.n_experts as usize;
    if n == 0 {
        return Layout::identity(0);
    }
    let mut placed = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);

    // Begin with the heaviest pair, so the chain starts inside a real cluster.
    let mut best = (0u32, 0u32, 0u32);
    for i in 0..n as u32 {
        for j in (i + 1)..n as u32 {
            let w = ca.co(i, j);
            if w > best.2 {
                best = (i, j, w);
            }
        }
    }
    let mut tail = if best.2 > 0 {
        order.push(best.0);
        order.push(best.1);
        placed[best.0 as usize] = true;
        placed[best.1 as usize] = true;
        best.1
    } else {
        // No co-activation anywhere: any order is as good as any other.
        order.push(0);
        placed[0] = true;
        0
    };
    while order.len() < n {
        // Affinity to the tail, tie-broken by affinity to the whole recent
        // window, so a zero-co-activation tail does not pick arbitrarily.
        let window_start = order.len().saturating_sub(4);
        let mut pick = None;
        let mut pick_score = (0u64, 0u64);
        for e in 0..n as u32 {
            if placed[e as usize] {
                continue;
            }
            let direct = u64::from(ca.co(tail, e));
            let window: u64 = order[window_start..]
                .iter()
                .map(|&o| u64::from(ca.co(o, e)))
                .sum();
            let score = (direct, window);
            if pick.is_none() || score > pick_score {
                pick = Some(e);
                pick_score = score;
            }
        }
        let e = pick.expect("an unplaced expert exists while order.len() < n");
        placed[e as usize] = true;
        order.push(e);
        tail = e;
    }
    Layout::from_expert_at(order).expect("greedy chain is a permutation")
}

/// Improve an order by local search against the real objective.
pub fn optimize(
    edges: &Edges,
    ca: &CoActivation,
    cm: &CostModel,
    cfg: &OptimizeConfig,
) -> OptimizeReport {
    let n = ca.n_experts as usize;
    let baseline_layout = Layout::identity(ca.n_experts);
    let baseline = evaluate(edges, &baseline_layout, cm);

    let mut layout = seed_by_coactivation(ca);
    let seeded = evaluate(edges, &layout, cm);

    // Candidate partners: the experts each expert co-fires with most. Swapping
    // toward a strong partner is what can shorten a run.
    let mut partners: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in 0..n as u32 {
        let mut v: Vec<(u32, u32)> = (0..n as u32)
            .filter(|&j| j != i)
            .map(|j| (ca.co(i, j), j))
            .filter(|(w, _)| *w > 0)
            .collect();
        v.sort_unstable_by_key(|&(w, _)| std::cmp::Reverse(w));
        v.truncate(cfg.candidates);
        partners.push(v.into_iter().map(|(_, j)| j).collect());
    }

    let mut scratch = Vec::with_capacity(edges.k);
    let mut affected: Vec<u32> = Vec::with_capacity(1024);
    let mut swaps = 0usize;
    let mut passes = 0usize;

    // The best layout seen under a *full* evaluation, so approximate candidate
    // scoring can never leave us worse off than where we started.
    let mut best_layout = layout.clone();
    let mut best_cost = seeded.seconds_per_token;

    for _ in 0..cfg.max_passes {
        passes += 1;
        for e in 0..n as u32 {
            let e_slot = layout.slot_of[e as usize] as usize;
            let mut best: Option<(f64, usize)> = None;
            for &p in &partners[e as usize] {
                let p_slot = layout.slot_of[p as usize] as usize;
                // Try placing e directly beside p, on either side.
                for target in [p_slot.saturating_sub(1), (p_slot + 1).min(n - 1)] {
                    if target == e_slot {
                        continue;
                    }
                    let delta = swap_delta(
                        edges,
                        &mut layout,
                        cm,
                        &mut scratch,
                        &mut affected,
                        cfg.max_delta_edges,
                        e_slot,
                        target,
                    );
                    let improves = match best {
                        None => true,
                        Some((bd, _)) => delta < bd,
                    };
                    if delta < -f64::EPSILON && improves {
                        best = Some((delta, target));
                    }
                }
            }
            if let Some((_, target)) = best {
                layout.swap_slots(e_slot, target);
                swaps += 1;
            }
        }
        // Confirm the pass against the true objective.
        let true_cost = evaluate(edges, &layout, cm).seconds_per_token;
        if true_cost < best_cost {
            let gain = best_cost - true_cost;
            let rel = gain / best_cost.abs().max(f64::MIN_POSITIVE);
            best_cost = true_cost;
            best_layout = layout.clone();
            if rel < cfg.min_improvement {
                break;
            }
        } else {
            // This pass did not help; the best known order is already retained.
            break;
        }
    }
    let layout = best_layout;

    let optimized = evaluate(edges, &layout, cm);
    OptimizeReport {
        layout,
        baseline,
        seeded,
        optimized,
        passes,
        swaps,
    }
}

/// Change in total cost from swapping the experts in slots `a` and `b`.
///
/// Only edges containing either expert can change, so the delta is computed
/// over just those. The layout is mutated and restored internally.
#[allow(clippy::too_many_arguments)]
fn swap_delta(
    edges: &Edges,
    layout: &mut Layout,
    cm: &CostModel,
    scratch: &mut Vec<u32>,
    affected: &mut Vec<u32>,
    max_edges: usize,
    a: usize,
    b: usize,
) -> f64 {
    let ea = layout.expert_at[a];
    let eb = layout.expert_at[b];

    // Union of affected edge indices, without allocating a set per call.
    let la = &edges.by_expert[ea as usize];
    let lb = &edges.by_expert[eb as usize];

    let mut i = 0usize;
    let mut j = 0usize;
    affected.clear();
    while i < la.len() || j < lb.len() {
        let pick = if j >= lb.len() {
            let v = la[i];
            i += 1;
            v
        } else if i >= la.len() {
            let v = lb[j];
            j += 1;
            v
        } else if la[i] < lb[j] {
            let v = la[i];
            i += 1;
            v
        } else if la[i] > lb[j] {
            let v = lb[j];
            j += 1;
            v
        } else {
            let v = la[i];
            i += 1;
            j += 1;
            v
        };
        affected.push(pick);
    }
    // Score every affected edge when there are few, otherwise a stride sample.
    let total = affected.len();
    let stride = if max_edges == 0 || total <= max_edges {
        1
    } else {
        total.div_ceil(max_edges)
    };
    let mut sampled = 0usize;

    let mut before = 0.0f64;
    let mut k = 0usize;
    while k < total {
        let idx = affected[k];
        let (bytes, reqs) = edge_shape(edges.edge(idx as usize), layout, cm, scratch);
        before += seconds_for(bytes, reqs, cm);
        sampled += 1;
        k += stride;
    }

    layout.swap_slots(a, b);
    let mut after = 0.0f64;
    let mut k = 0usize;
    while k < total {
        let idx = affected[k];
        let (bytes, reqs) = edge_shape(edges.edge(idx as usize), layout, cm, scratch);
        after += seconds_for(bytes, reqs, cm);
        k += stride;
    }
    layout.swap_slots(a, b);

    if sampled == 0 {
        return 0.0;
    }
    // Scale the sampled difference back to the whole affected set.
    let scale = total as f64 / sampled as f64;
    ((after - before) * scale) / edges.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmx_probe::{Cell, Surface};

    fn measured_surface() -> Surface {
        // The surface measured on the development machine (GB/s -> B/s).
        let raw: &[(u64, usize, f64)] = &[
            (64 << 10, 1, 0.12),
            (64 << 10, 4, 0.41),
            (64 << 10, 8, 0.82),
            (64 << 10, 16, 1.36),
            (256 << 10, 1, 0.26),
            (256 << 10, 4, 1.24),
            (256 << 10, 8, 2.15),
            (256 << 10, 16, 2.02),
            (1 << 20, 1, 0.64),
            (1 << 20, 4, 1.70),
            (1 << 20, 8, 2.21),
            (1 << 20, 16, 2.40),
            (2 << 20, 1, 0.79),
            (2 << 20, 4, 2.10),
            (2 << 20, 8, 2.41),
            (2 << 20, 16, 2.34),
        ];
        Surface {
            note: "development machine".into(),
            cache_bypassed: true,
            cells: raw
                .iter()
                .map(|&(b, q, g)| Cell {
                    blob_bytes: b,
                    queue_depth: q,
                    bytes_per_sec: g * 1e9,
                })
                .collect(),
        }
    }

    fn model(slice: u64) -> CostModel {
        CostModel {
            slice_bytes: slice,
            tensors_per_expert: 3,
            merge_gap_slices: 0,
            queue_depth: 8,
            surface: measured_surface(),
        }
    }

    #[test]
    fn layout_rejects_non_permutations() {
        assert!(Layout::from_expert_at(vec![0, 1, 1]).is_none());
        assert!(Layout::from_expert_at(vec![0, 5]).is_none());
        assert!(Layout::from_expert_at(vec![2, 0, 1]).is_some());
    }

    #[test]
    fn swap_keeps_the_two_indices_consistent() {
        let mut l = Layout::identity(4);
        l.swap_slots(0, 3);
        assert_eq!(l.expert_at(), &[3, 1, 2, 0]);
        for (e, &s) in l.slot_of().iter().enumerate() {
            assert_eq!(l.expert_at()[s as usize], e as u32);
        }
    }

    #[test]
    fn adjacent_experts_cost_fewer_requests_than_scattered() {
        let cm = model(64 << 10);
        // One token selecting slots 0..4 versus slots spread far apart.
        let t = Trace {
            n_layers: 1,
            n_experts: 16,
            top_k: 4,
            selections: vec![0, 1, 2, 3],
        };
        let edges = Edges::from_trace(&t, 0);
        let adjacent = evaluate(&edges, &Layout::identity(16), &cm);

        let spread =
            Layout::from_expert_at(vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
                .unwrap();
        let scattered = evaluate(&edges, &spread, &cm);

        assert!(
            adjacent.requests_per_token < scattered.requests_per_token,
            "adjacent {:?} should need fewer requests than scattered {:?}",
            adjacent,
            scattered
        );
    }

    #[test]
    fn merge_gap_trades_bytes_for_requests() {
        let mut cm = model(64 << 10);
        // Slots 0 and 2 needed: one unused slot between them.
        let t = Trace {
            n_layers: 1,
            n_experts: 8,
            top_k: 2,
            selections: vec![0, 2],
        };
        let edges = Edges::from_trace(&t, 0);
        let l = Layout::identity(8);

        cm.merge_gap_slices = 0;
        let split = evaluate(&edges, &l, &cm);
        cm.merge_gap_slices = 1;
        let merged = evaluate(&edges, &l, &cm);

        assert!(merged.requests_per_token < split.requests_per_token);
        assert!(
            merged.bytes_per_token > split.bytes_per_token,
            "merging must over-read"
        );
    }

    #[test]
    fn optimizer_beats_a_deliberately_shuffled_order() {
        // Clustered routing, but the checkpoint interleaves clusters so that
        // co-firing experts start maximally far apart.
        let n = 32u32;
        let base = Trace::synthetic(1, n, 4, 3000, 4, 0.95, 21);
        // Relabel experts so cluster members are strided across the axis.
        let stride = |e: u32| -> u32 { (e % 4) * 8 + (e / 4) };
        let shuffled = Trace {
            n_layers: 1,
            n_experts: n,
            top_k: 4,
            selections: base.selections.iter().map(|&e| stride(e)).collect(),
        };
        let edges = Edges::from_trace(&shuffled, 0);
        let ca = CoActivation::from_trace(&shuffled, 0);
        let cm = model(64 << 10);
        let rep = optimize(&edges, &ca, &cm, &OptimizeConfig::default());

        assert!(
            rep.optimized.seconds_per_token <= rep.baseline.seconds_per_token,
            "optimiser must not regress: {:?} vs {:?}",
            rep.optimized,
            rep.baseline
        );
        assert!(
            rep.optimized.requests_per_token < rep.baseline.requests_per_token,
            "clustered routing should coalesce into fewer requests: {} -> {}",
            rep.baseline.requests_per_token,
            rep.optimized.requests_per_token
        );
        assert!(rep.layout.expert_at().len() == n as usize);
    }

    #[test]
    fn random_routing_offers_nothing_to_optimize() {
        // With no locality there is no layout win, and the tool should say so
        // rather than manufacture one.
        let n = 32u32;
        let t = Trace::synthetic(1, n, 4, 3000, 1, 0.0, 5);
        let edges = Edges::from_trace(&t, 0);
        let ca = CoActivation::from_trace(&t, 0);
        let rep = optimize(&edges, &ca, &model(64 << 10), &OptimizeConfig::default());
        assert!(
            !rep.worth_repacking(1.05),
            "random routing should not be judged worth repacking (speedup {:.3})",
            rep.speedup()
        );
    }

    #[test]
    fn seed_is_a_valid_permutation() {
        let t = Trace::synthetic(1, 40, 6, 1500, 5, 0.9, 3);
        let ca = CoActivation::from_trace(&t, 0);
        let l = seed_by_coactivation(&ca);
        let mut seen = l.expert_at().to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..40u32).collect::<Vec<_>>());
    }

    #[test]
    fn swap_delta_agrees_with_full_reevaluation() {
        let t = Trace::synthetic(1, 24, 4, 800, 3, 0.85, 13);
        let edges = Edges::from_trace(&t, 0);
        let cm = model(64 << 10);
        let mut layout = Layout::identity(24);
        let mut scratch = Vec::new();

        let before = evaluate(&edges, &layout, &cm).seconds_per_token;
        let mut affected = Vec::new();
        // max_edges = 0 disables sampling, so the delta must be exact.
        let delta = swap_delta(
            &edges,
            &mut layout,
            &cm,
            &mut scratch,
            &mut affected,
            0,
            3,
            17,
        );
        layout.swap_slots(3, 17);
        let after = evaluate(&edges, &layout, &cm).seconds_per_token;

        assert!(
            ((before + delta) - after).abs() < 1e-12,
            "incremental delta {delta} disagrees: {before} + delta != {after}"
        );
    }
}
