// SPDX-License-Identifier: GPL-2.0-or-later
//! Router lookahead.
//!
//! # Why prediction is the whole game
//!
//! A device's read bandwidth depends far more on how many requests are in flight
//! than on anything a file layout can control — measured at 6-8x between queue
//! depth 1 and 8 on the development machine. But you cannot have eight reads
//! outstanding for experts you have not been told about yet. The router for
//! layer *n* runs only after layer *n-1* has finished, so by the time the true
//! answer exists there is no time left to hide the fetch behind compute.
//!
//! Prediction is what buys that time, and it is therefore the mechanism by which
//! the largest available speedup is actually claimed. A wrong prediction is
//! worse than none: it spends bandwidth *and* still stalls.
//!
//! # What is implemented here, and what the literature achieves
//!
//! The predictors here are **training-free**. They use only the routing history
//! a running engine already has, so they work on any checkpoint with no
//! calibration step:
//!
//! * [`Predictor::Sticky`] — reuse the previous token's selection for this
//!   layer. Exploits the temporal locality that makes expert caches work at all.
//! * [`Predictor::Markov`] — learn, per layer, which experts tend to follow
//!   which. Uses the previous *layer's* selection to predict this one.
//! * [`Predictor::Frequency`] — always guess the globally hottest experts. A
//!   deliberately weak baseline: any predictor that cannot beat it is worthless.
//! * [`Predictor::StickyMarkov`] — union of Sticky and Markov, which trades
//!   fetching a few more experts for materially better recall.
//!
//! Trained predictors do better and it would be dishonest to imply otherwise.
//! Colibri's PILOT reports 71.6% one layer ahead, improved to 76.7% by folding
//! the shared expert into the residual first. Pre-attention learned routers
//! reach 93.0% on DeepSeek-V2-Lite, 94.7% on Qwen3-30B and 97.6% on
//! Phi-mini-MoE. Those need a trained head; these do not. [`evaluate`] measures
//! whichever you use on your own trace, so the comparison is empirical.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pmx_trace::Trace;

/// A training-free expert predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    /// Predict the globally most-selected experts. Weak baseline.
    Frequency,
    /// Predict whatever this layer selected for the previous token.
    Sticky,
    /// Predict from the previous layer's selection, using learned transitions.
    Markov,
    /// Union of [`Predictor::Sticky`] and [`Predictor::Markov`].
    StickyMarkov,
}

impl Predictor {
    /// Every predictor.
    pub const ALL: [Predictor; 4] = [
        Predictor::Frequency,
        Predictor::Sticky,
        Predictor::Markov,
        Predictor::StickyMarkov,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Predictor::Frequency => "frequency",
            Predictor::Sticky => "sticky",
            Predictor::Markov => "markov",
            Predictor::StickyMarkov => "sticky+markov",
        }
    }

    /// Parse a predictor name.
    pub fn parse(s: &str) -> Option<Predictor> {
        Some(match s {
            "frequency" => Predictor::Frequency,
            "sticky" => Predictor::Sticky,
            "markov" => Predictor::Markov,
            "sticky+markov" | "stickymarkov" => Predictor::StickyMarkov,
            _ => return None,
        })
    }
}

/// Statistics learned from a trace prefix.
///
/// Everything here is observable by a running engine without any training: it is
/// just counting what the router has already done.
#[derive(Debug, Clone)]
pub struct RoutingModel {
    n_layers: u32,
    n_experts: u32,
    top_k: usize,
    /// `freq[layer][expert]`: selections observed.
    freq: Vec<Vec<u64>>,
    /// `trans[layer][prev_expert]` -> counts of experts chosen at `layer` when
    /// `prev_expert` was selected at `layer - 1`.
    trans: Vec<Vec<Vec<u32>>>,
    /// Hottest experts per layer, descending. Cached for the frequency predictor.
    hot: Vec<Vec<u32>>,
}

impl RoutingModel {
    /// Learn from the first `tokens` tokens of `trace`.
    ///
    /// Splitting a trace into a fit prefix and an evaluation suffix is what keeps
    /// [`evaluate`] honest — a predictor scored on the tokens it was fitted to
    /// would report a number no deployment could reproduce.
    pub fn fit(trace: &Trace, tokens: usize) -> Self {
        let nl = trace.n_layers as usize;
        let ne = trace.n_experts as usize;
        let k = trace.top_k as usize;
        let tokens = tokens.min(trace.n_tokens());

        let mut freq = vec![vec![0u64; ne]; nl];
        let mut trans = vec![Vec::new(); nl];
        // Layer 0 has no predecessor, so it gets no transition table.
        for t in trans.iter_mut().skip(1) {
            *t = vec![vec![0u32; ne]; ne];
        }

        for t in 0..tokens {
            for l in 0..nl {
                let cur = trace.selection(t, l as u32);
                for &e in cur {
                    freq[l][e as usize] += 1;
                }
                if l > 0 {
                    let prev = trace.selection(t, l as u32 - 1);
                    for &p in prev {
                        for &e in cur {
                            let c = &mut trans[l][p as usize][e as usize];
                            *c = c.saturating_add(1);
                        }
                    }
                }
            }
        }

        let hot = freq
            .iter()
            .map(|f| {
                let mut idx: Vec<u32> = (0..ne as u32).collect();
                idx.sort_unstable_by_key(|&e| std::cmp::Reverse(f[e as usize]));
                idx
            })
            .collect();

        RoutingModel {
            n_layers: trace.n_layers,
            n_experts: trace.n_experts,
            top_k: k,
            freq,
            trans,
            hot,
        }
    }

    /// Experts per token per layer in the fitted trace.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// MoE layers covered.
    pub fn n_layers(&self) -> u32 {
        self.n_layers
    }

    /// Experts per layer.
    pub fn n_experts(&self) -> u32 {
        self.n_experts
    }

    /// Selections observed for one expert during fitting.
    ///
    /// This is the frequency a residency plan and a cost-aware cache both key
    /// off, so it is exposed rather than kept private.
    pub fn frequency(&self, layer: u32, expert: u32) -> u64 {
        self.freq
            .get(layer as usize)
            .and_then(|f| f.get(expert as usize))
            .copied()
            .unwrap_or(0)
    }

    /// Experts at `layer`, hottest first.
    pub fn hottest(&self, layer: u32) -> &[u32] {
        &self.hot[layer as usize]
    }

    /// Share of selections at `layer` taken by its `n` hottest experts.
    pub fn skew(&self, layer: u32, n: usize) -> f64 {
        let f = &self.freq[layer as usize];
        let total: u64 = f.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let head: u64 = self.hot[layer as usize]
            .iter()
            .take(n)
            .map(|&e| f[e as usize])
            .sum();
        head as f64 / total as f64
    }

    /// Predict `budget` experts for `layer`, given context.
    ///
    /// `prev_layer` is the selection made at `layer - 1` of the *current* token —
    /// available in a real engine, since layers run in order. `prev_token` is
    /// this layer's selection for the previous token. Either may be empty at a
    /// boundary, and the predictor degrades gracefully.
    pub fn predict(
        &self,
        which: Predictor,
        layer: u32,
        prev_layer: &[u32],
        prev_token: &[u32],
        budget: usize,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        let l = layer as usize;
        match which {
            Predictor::Frequency => {
                out.extend(self.hot[l].iter().take(budget).copied());
            }
            Predictor::Sticky => {
                out.extend(prev_token.iter().take(budget).copied());
                self.pad_with_hot(l, budget, out);
            }
            Predictor::Markov => {
                self.markov_into(l, prev_layer, budget, out);
                self.pad_with_hot(l, budget, out);
            }
            Predictor::StickyMarkov => {
                // The previous token's choice is the single strongest signal, so
                // it goes in first and Markov fills the remaining budget.
                for &e in prev_token.iter().take(budget) {
                    if !out.contains(&e) {
                        out.push(e);
                    }
                }
                if out.len() < budget {
                    let mut m = Vec::new();
                    self.markov_into(l, prev_layer, budget, &mut m);
                    for e in m {
                        if out.len() >= budget {
                            break;
                        }
                        if !out.contains(&e) {
                            out.push(e);
                        }
                    }
                }
                self.pad_with_hot(l, budget, out);
            }
        }
        out.truncate(budget);
    }

    fn markov_into(&self, l: usize, prev_layer: &[u32], budget: usize, out: &mut Vec<u32>) {
        if l == 0 || prev_layer.is_empty() || self.trans[l].is_empty() {
            return;
        }
        // Score each candidate by summed transition count from the experts that
        // actually fired at the previous layer.
        let ne = self.n_experts as usize;
        let mut score = vec![0u64; ne];
        for &p in prev_layer {
            let row = &self.trans[l][p as usize];
            for (e, c) in row.iter().enumerate() {
                score[e] += u64::from(*c);
            }
        }
        let mut idx: Vec<u32> = (0..ne as u32).filter(|&e| score[e as usize] > 0).collect();
        idx.sort_unstable_by_key(|&e| std::cmp::Reverse(score[e as usize]));
        out.extend(idx.into_iter().take(budget));
    }

    fn pad_with_hot(&self, l: usize, budget: usize, out: &mut Vec<u32>) {
        if out.len() >= budget {
            return;
        }
        for &e in &self.hot[l] {
            if out.len() >= budget {
                break;
            }
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
}

/// How well a predictor did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    /// Predictions scored.
    pub samples: u64,
    /// Mean share of truly-selected experts that were predicted.
    ///
    /// This is "ready recall": the fraction of the router's real choices that a
    /// prefetcher would already have had in flight.
    pub recall: f64,
    /// Mean share of predicted experts that were actually used.
    ///
    /// The complement is wasted bandwidth, which is why recall alone is not a
    /// sufficient score.
    pub precision: f64,
    /// Experts predicted per step.
    pub budget: usize,
}

impl Accuracy {
    /// Experts fetched per expert usefully obtained. 1.0 is perfect.
    pub fn fetch_amplification(&self) -> f64 {
        if self.precision <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.precision
        }
    }
}

/// Score `which` on `trace`, fitting on the first `fit_tokens` and evaluating on
/// the rest.
///
/// `budget` is how many experts the predictor may name; passing more than
/// `top_k` trades bandwidth for recall, which is often the right trade when a
/// miss costs a stall.
pub fn evaluate(
    trace: &Trace,
    which: Predictor,
    fit_tokens: usize,
    budget: usize,
) -> Option<Accuracy> {
    let n = trace.n_tokens();
    if n == 0 || fit_tokens >= n {
        return None;
    }
    let model = RoutingModel::fit(trace, fit_tokens);
    let mut pred = Vec::new();
    let mut samples = 0u64;
    let mut recall_acc = 0.0f64;
    let mut prec_acc = 0.0f64;

    for t in fit_tokens..n {
        for l in 0..trace.n_layers {
            let truth = trace.selection(t, l);
            let prev_layer: &[u32] = if l > 0 {
                trace.selection(t, l - 1)
            } else {
                &[]
            };
            let prev_token: &[u32] = if t > 0 {
                trace.selection(t - 1, l)
            } else {
                &[]
            };
            model.predict(which, l, prev_layer, prev_token, budget, &mut pred);

            let hit = truth.iter().filter(|e| pred.contains(e)).count();
            recall_acc += hit as f64 / truth.len().max(1) as f64;
            prec_acc += hit as f64 / pred.len().max(1) as f64;
            samples += 1;
        }
    }
    if samples == 0 {
        return None;
    }
    Some(Accuracy {
        samples,
        recall: recall_acc / samples as f64,
        precision: prec_acc / samples as f64,
        budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trace with both within-token co-activation and across-token
    /// persistence. Persistence is the part that matters here: with zero
    /// persistence there is no temporal structure and no predictor can do better
    /// than guessing the hottest experts.
    fn clustered(seed: u64) -> Trace {
        let mut t = Trace::synthetic_cfg(&pmx_trace::SynthConfig {
            n_layers: 4,
            n_experts: 64,
            top_k: 6,
            tokens: 4000,
            clusters: 8,
            locality: 0.9,
            persistence: 0.7,
            layer_coupling: 0.5,
            seed,
        });
        t.scatter_labels(0xBEEF);
        t
    }

    #[test]
    fn union_only_helps_above_the_top_k_budget() {
        // At budget == top_k the sticky prediction fills the whole budget, so the
        // union has no room to contribute and must tie. Worth pinning down: it is
        // the reason a prefetch budget above top_k is what buys recall.
        let t = clustered(17);
        let k = t.top_k as usize;
        let s = evaluate(&t, Predictor::Sticky, 2000, k).unwrap();
        let sm = evaluate(&t, Predictor::StickyMarkov, 2000, k).unwrap();
        assert!(
            (s.recall - sm.recall).abs() < 1e-12,
            "{:.4} vs {:.4}",
            s.recall,
            sm.recall
        );

        let s2 = evaluate(&t, Predictor::Sticky, 2000, k * 2).unwrap();
        let sm2 = evaluate(&t, Predictor::StickyMarkov, 2000, k * 2).unwrap();
        assert!(
            sm2.recall > s2.recall,
            "at double budget the union should win: {:.3} vs {:.3}",
            sm2.recall,
            s2.recall
        );
    }

    #[test]
    fn model_exposes_the_skew_a_plan_depends_on() {
        let t = clustered(19);
        let m = RoutingModel::fit(&t, 3000);
        assert_eq!(m.n_experts(), 64);
        assert_eq!(m.n_layers(), 4);
        // Clustered routing must beat the uniform share of 16/64.
        assert!(m.skew(0, 16) > 0.25, "skew {:.3}", m.skew(0, 16));
        assert_eq!(m.hottest(0).len(), 64);
        let hottest = m.hottest(0)[0];
        assert!(m.frequency(0, hottest) > 0);
    }

    #[test]
    fn names_round_trip() {
        for p in Predictor::ALL {
            assert_eq!(Predictor::parse(p.name()), Some(p));
        }
        assert_eq!(Predictor::parse("nope"), None);
    }

    #[test]
    fn predictions_respect_the_budget_and_are_distinct() {
        let t = clustered(3);
        let m = RoutingModel::fit(&t, 2000);
        let mut out = Vec::new();
        for which in Predictor::ALL {
            for l in 0..t.n_layers {
                m.predict(
                    which,
                    l,
                    t.selection(5, l.saturating_sub(1)),
                    t.selection(4, l),
                    10,
                    &mut out,
                );
                assert!(out.len() <= 10, "{}: budget exceeded", which.name());
                let mut v = out.clone();
                v.sort_unstable();
                v.dedup();
                assert_eq!(
                    v.len(),
                    out.len(),
                    "{}: duplicate predictions",
                    which.name()
                );
                assert!(
                    out.iter().all(|e| *e < t.n_experts),
                    "{}: expert id out of range",
                    which.name()
                );
            }
        }
    }

    #[test]
    fn budget_is_filled_even_with_no_context() {
        let t = clustered(4);
        let m = RoutingModel::fit(&t, 1000);
        let mut out = Vec::new();
        for which in Predictor::ALL {
            m.predict(which, 0, &[], &[], 8, &mut out);
            assert_eq!(out.len(), 8, "{} left the budget unfilled", which.name());
        }
    }

    #[test]
    fn every_predictor_beats_the_frequency_baseline() {
        // If a predictor cannot beat "always guess the hottest experts", it is
        // not carrying any information and should not be shipped.
        let t = clustered(7);
        let base = evaluate(&t, Predictor::Frequency, 2000, 6).unwrap();
        for which in [
            Predictor::Sticky,
            Predictor::Markov,
            Predictor::StickyMarkov,
        ] {
            let a = evaluate(&t, which, 2000, 6).unwrap();
            assert!(
                a.recall > base.recall,
                "{}: recall {:.3} did not beat the frequency baseline {:.3}",
                which.name(),
                a.recall,
                base.recall
            );
        }
    }

    #[test]
    fn recall_rises_with_budget_and_precision_falls() {
        // The core trade: naming more experts catches more of the router's real
        // choices, at the cost of fetching some that go unused.
        let t = clustered(9);
        let small = evaluate(&t, Predictor::StickyMarkov, 2000, 6).unwrap();
        let large = evaluate(&t, Predictor::StickyMarkov, 2000, 18).unwrap();
        assert!(
            large.recall > small.recall,
            "recall {:.3} -> {:.3} did not improve with budget",
            small.recall,
            large.recall
        );
        assert!(
            large.precision < small.precision,
            "precision {:.3} -> {:.3} should fall as budget grows",
            small.precision,
            large.precision
        );
        assert!(large.fetch_amplification() > small.fetch_amplification());
    }

    #[test]
    fn random_routing_leaves_nothing_to_predict() {
        // With no locality, no training-free predictor can beat guessing the
        // hottest experts. Reporting that honestly is the point.
        let t = Trace::synthetic(2, 64, 6, 4000, 1, 0.0, 5);
        let base = evaluate(&t, Predictor::Frequency, 2000, 6).unwrap();
        let sticky = evaluate(&t, Predictor::Sticky, 2000, 6).unwrap();
        // Both should sit near the chance level of 6/64.
        let chance = 6.0 / 64.0;
        assert!(
            (base.recall - chance).abs() < 0.05,
            "frequency recall {:.3} vs chance {chance:.3}",
            base.recall
        );
        assert!(
            sticky.recall < chance + 0.05,
            "sticky recall {:.3} should not exceed chance on random routing",
            sticky.recall
        );
    }

    #[test]
    fn evaluation_refuses_to_score_on_the_fitting_data() {
        let t = clustered(11);
        // fit_tokens >= n leaves no held-out tokens.
        assert!(evaluate(&t, Predictor::Markov, t.n_tokens(), 6).is_none());
        assert!(evaluate(&t, Predictor::Markov, t.n_tokens() + 10, 6).is_none());
    }

    #[test]
    fn stickymarkov_is_at_least_as_good_as_its_parts() {
        let t = clustered(13);
        let s = evaluate(&t, Predictor::Sticky, 2000, 8).unwrap();
        let m = evaluate(&t, Predictor::Markov, 2000, 8).unwrap();
        let sm = evaluate(&t, Predictor::StickyMarkov, 2000, 8).unwrap();
        assert!(
            sm.recall >= s.recall.max(m.recall) - 1e-9,
            "union recall {:.3} below sticky {:.3} / markov {:.3}",
            sm.recall,
            s.recall,
            m.recall
        );
    }
}
