// SPDX-License-Identifier: GPL-2.0-or-later
//! Routing traces and the co-activation statistics derived from them.
//!
//! A trace records, for every generated token and every MoE layer, which
//! experts the router selected. Two things are extracted from it:
//!
//! * **Access frequency** per expert — extremely skewed in practice, which is
//!   what makes a small resident cache effective and what drives bit allocation.
//! * **Co-activation** — how often two experts are selected for the *same*
//!   token. This is the signal the layout partitioner optimises against, and it
//!   is the reason locality exists to be exploited at all: expert selections
//!   for adjacent tokens overlap at roughly twice the rate random routing would
//!   produce.
//!
//! The binary format is deliberately trivial so that patching an engine to emit
//! one is a small change. A line-oriented text form is accepted too.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

/// File magic for the binary trace format.
pub const MAGIC: [u8; 8] = *b"PMXTRACE";
/// Current binary format version.
pub const VERSION: u32 = 2;

/// The original format, which had no prefill/decode boundary. Still read; such
/// a file loads with [`Trace::prefill_tokens`] at zero.
pub const VERSION_V1: u32 = 1;

/// Anything that can go wrong handling a trace.
#[derive(Debug)]
pub enum TraceError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// Wrong file magic.
    BadMagic,
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// Header described a shape that cannot be right.
    BadHeader(String),
    /// The record body was not a whole number of selections.
    Ragged {
        /// Selection values found.
        values: usize,
        /// Values expected per token.
        per_token: usize,
    },
    /// An expert id fell outside the declared range.
    ExpertOutOfRange {
        /// The offending id.
        id: u32,
        /// Declared expert count.
        n_experts: u32,
    },
    /// A text trace line could not be parsed.
    BadTextLine {
        /// 1-based line number.
        line: usize,
        /// What went wrong.
        why: String,
    },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Io(e) => write!(f, "io error: {e}"),
            TraceError::BadMagic => write!(f, "not a pmx trace file"),
            TraceError::UnsupportedVersion(v) => write!(f, "unsupported trace version {v}"),
            TraceError::BadHeader(m) => write!(f, "bad trace header: {m}"),
            TraceError::Ragged { values, per_token } => write!(
                f,
                "trace body holds {values} selections, not a multiple of {per_token} per token"
            ),
            TraceError::ExpertOutOfRange { id, n_experts } => {
                write!(f, "expert id {id} outside declared range 0..{n_experts}")
            }
            TraceError::BadTextLine { line, why } => write!(f, "line {line}: {why}"),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<std::io::Error> for TraceError {
    fn from(e: std::io::Error) -> Self {
        TraceError::Io(e)
    }
}

/// A captured routing trace.
///
/// `selections` is laid out token-major then layer-major:
/// `selections[(token * n_layers + layer) * top_k + i]`.
#[derive(Debug, Clone)]
pub struct Trace {
    /// Number of MoE layers.
    pub n_layers: u32,
    /// Experts per layer.
    pub n_experts: u32,
    /// Experts selected per token per layer.
    pub top_k: u32,
    /// Flat selection array.
    pub selections: Vec<u32>,
    /// How many leading tokens belong to a prefill burst rather than to decode.
    ///
    /// Zero means the trace is undifferentiated — which is what every synthetic
    /// trace was before this existed, and the reason segmented cache policies
    /// could not be told apart from plain LRU. Prefill reads many tokens at
    /// once and therefore touches a wide set of experts; decode reads one token
    /// at a time and returns to a narrow set. A recency-only policy evicts the
    /// narrow set to make room for the wide one, then misses on all of it.
    /// Expressing the boundary is what makes that failure observable.
    pub prefill_tokens: usize,
}

impl Trace {
    /// Create an empty trace with the given shape.
    pub fn new(n_layers: u32, n_experts: u32, top_k: u32) -> Self {
        Trace {
            n_layers,
            n_experts,
            top_k,
            selections: Vec::new(),
            prefill_tokens: 0,
        }
    }

    /// Selections per token, across all layers.
    pub fn per_token(&self) -> usize {
        self.n_layers as usize * self.top_k as usize
    }

    /// Number of tokens in the trace.
    pub fn n_tokens(&self) -> usize {
        self.selections
            .len()
            .checked_div(self.per_token())
            .unwrap_or(0)
    }

    /// The experts selected for `token` at `layer`.
    pub fn selection(&self, token: usize, layer: u32) -> &[u32] {
        let k = self.top_k as usize;
        let base = (token * self.n_layers as usize + layer as usize) * k;
        &self.selections[base..base + k]
    }

    /// Iterate every token's selection at one layer. These are the hyperedges
    /// the partitioner minimises against.
    pub fn layer_edges(&self, layer: u32) -> impl Iterator<Item = &[u32]> + '_ {
        (0..self.n_tokens()).map(move |t| self.selection(t, layer))
    }

    /// Tokens that are decode rather than prefill.
    pub fn decode_tokens(&self) -> usize {
        self.n_tokens().saturating_sub(self.prefill_tokens)
    }

    /// Selections belonging to the prefill burst. Empty when there is no phase
    /// boundary.
    pub fn prefill_selections(&self) -> &[u32] {
        let end = (self.prefill_tokens * self.per_token()).min(self.selections.len());
        &self.selections[..end]
    }

    /// Selections belonging to decode.
    ///
    /// This is the part a cache is judged on. Prefill's misses are unavoidable —
    /// nothing is resident yet — so counting them dilutes the difference between
    /// policies with a constant every policy pays.
    pub fn decode_selections(&self) -> &[u32] {
        let start = (self.prefill_tokens * self.per_token()).min(self.selections.len());
        &self.selections[start..]
    }

    /// Check every id is in range and the body is not ragged.
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.n_layers == 0 || self.n_experts == 0 || self.top_k == 0 {
            return Err(TraceError::BadHeader(format!(
                "n_layers={}, n_experts={}, top_k={} must all be non-zero",
                self.n_layers, self.n_experts, self.top_k
            )));
        }
        if self.top_k > self.n_experts {
            return Err(TraceError::BadHeader(format!(
                "top_k {} exceeds n_experts {}",
                self.top_k, self.n_experts
            )));
        }
        let pt = self.per_token();
        if self.selections.len() % pt != 0 {
            return Err(TraceError::Ragged {
                values: self.selections.len(),
                per_token: pt,
            });
        }
        if self.prefill_tokens > self.selections.len() / pt {
            return Err(TraceError::BadHeader(format!(
                "prefill_tokens {} exceeds the {} tokens present",
                self.prefill_tokens,
                self.selections.len() / pt
            )));
        }
        for &e in &self.selections {
            if e >= self.n_experts {
                return Err(TraceError::ExpertOutOfRange {
                    id: e,
                    n_experts: self.n_experts,
                });
            }
        }
        Ok(())
    }

    /// Write the binary form.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), TraceError> {
        self.validate()?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&self.n_layers.to_le_bytes())?;
        w.write_all(&self.n_experts.to_le_bytes())?;
        w.write_all(&self.top_k.to_le_bytes())?;
        w.write_all(&(self.n_tokens() as u64).to_le_bytes())?;
        w.write_all(&(self.prefill_tokens as u64).to_le_bytes())?;
        for v in &self.selections {
            w.write_all(&v.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    /// Read the binary form, validating as it goes.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, TraceError> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(TraceError::BadMagic);
        }
        let mut u32b = [0u8; 4];
        let mut rd_u32 = |f: &mut dyn Read| -> Result<u32, TraceError> {
            f.read_exact(&mut u32b)?;
            Ok(u32::from_le_bytes(u32b))
        };
        let version = rd_u32(&mut f)?;
        if version != VERSION && version != VERSION_V1 {
            return Err(TraceError::UnsupportedVersion(version));
        }
        let n_layers = rd_u32(&mut f)?;
        let n_experts = rd_u32(&mut f)?;
        let top_k = rd_u32(&mut f)?;
        let mut u64b = [0u8; 8];
        f.read_exact(&mut u64b)?;
        let n_tokens = u64::from_le_bytes(u64b);
        // v1 predates the phase boundary. Reading it as all-decode is the
        // truthful interpretation: those traces really were undifferentiated.
        let prefill_tokens = if version == VERSION_V1 {
            0
        } else {
            f.read_exact(&mut u64b)?;
            u64::from_le_bytes(u64b)
        };
        if prefill_tokens > n_tokens {
            return Err(TraceError::BadHeader(format!(
                "header declares {prefill_tokens} prefill tokens of {n_tokens} total"
            )));
        }

        let per_token = n_layers as u64 * top_k as u64;
        let total = n_tokens
            .checked_mul(per_token)
            .ok_or_else(|| TraceError::BadHeader("selection count overflows".into()))?;
        // Bound the allocation by what the file can actually contain.
        let mut body = Vec::new();
        f.read_to_end(&mut body)?;
        if body.len() as u64 != total * 4 {
            return Err(TraceError::BadHeader(format!(
                "header declares {total} selections ({} bytes) but body holds {}",
                total * 4,
                body.len()
            )));
        }
        let selections = body
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let t = Trace {
            n_layers,
            n_experts,
            top_k,
            selections,
            prefill_tokens: prefill_tokens as usize,
        };
        t.validate()?;
        Ok(t)
    }

    /// Parse the line-oriented text form.
    ///
    /// Each line is `<token> <layer> <expert>...`, whitespace separated. Blank
    /// lines and lines beginning with `#` are ignored. Lines may arrive in any
    /// order; the shape is inferred from the maxima observed.
    pub fn from_text(text: &str) -> Result<Self, TraceError> {
        let mut rows: Vec<(usize, u32, Vec<u32>)> = Vec::new();
        let mut max_layer = 0u32;
        let mut max_expert = 0u32;
        let mut top_k = 0usize;
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let bad = |why: &str| TraceError::BadTextLine {
                line: i + 1,
                why: why.to_string(),
            };
            let token: usize = it
                .next()
                .ok_or_else(|| bad("missing token index"))?
                .parse()
                .map_err(|_| bad("token index is not a number"))?;
            let layer: u32 = it
                .next()
                .ok_or_else(|| bad("missing layer index"))?
                .parse()
                .map_err(|_| bad("layer index is not a number"))?;
            let mut experts = Vec::new();
            for tok in it {
                let e: u32 = tok.parse().map_err(|_| bad("expert id is not a number"))?;
                experts.push(e);
            }
            if experts.is_empty() {
                return Err(bad("no expert ids on line"));
            }
            max_layer = max_layer.max(layer);
            max_expert = max_expert.max(*experts.iter().max().unwrap());
            top_k = top_k.max(experts.len());
            rows.push((token, layer, experts));
        }
        if rows.is_empty() {
            return Err(TraceError::BadHeader(
                "text trace contained no records".into(),
            ));
        }
        let n_layers = max_layer + 1;
        let n_experts = max_expert + 1;
        let n_tokens = rows.iter().map(|(t, _, _)| *t).max().unwrap() + 1;

        let mut selections = vec![u32::MAX; n_tokens * n_layers as usize * top_k];
        for (t, l, es) in &rows {
            let base = (*t * n_layers as usize + *l as usize) * top_k;
            for (j, e) in es.iter().enumerate() {
                selections[base + j] = *e;
            }
            // Short rows are padded by repeating the last id, so the record stays
            // rectangular without inventing a selection that never happened.
            for j in es.len()..top_k {
                selections[base + j] = *es.last().unwrap();
            }
        }
        // Any (token, layer) never mentioned is filled with expert 0 repeated;
        // flag it rather than silently fabricating routing.
        if selections.contains(&u32::MAX) {
            return Err(TraceError::BadHeader(
                "text trace is missing some (token, layer) pairs; every layer must be recorded for every token"
                    .into(),
            ));
        }
        let t = Trace {
            n_layers,
            n_experts,
            top_k: top_k as u32,
            selections,
            prefill_tokens: 0,
        };
        t.validate()?;
        Ok(t)
    }

    /// Generate a synthetic trace from an explicit configuration.
    ///
    /// Prefer this over [`Trace::synthetic`] when the trace will be used to
    /// evaluate anything that depends on *temporal* structure. See
    /// [`SynthConfig::persistence`].
    pub fn synthetic_cfg(cfg: &SynthConfig) -> Self {
        let mut x = cfg.seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n_experts = cfg.n_experts;
        let clusters = cfg.clusters.max(1).min(n_experts);
        let per_cluster = (n_experts / clusters).max(1);
        let top_k = cfg.top_k as usize;
        // Prefill runs ahead of decode, so the trace is longer than `tokens`.
        let total_tokens = cfg.prefill_tokens + cfg.tokens;
        let mut selections = Vec::with_capacity(total_tokens * cfg.n_layers as usize * top_k);
        // Cluster carried between tokens, per layer, to create persistence.
        let mut last_cluster = vec![0u32; cfg.n_layers as usize];
        let mut have_last = false;

        for token in 0..total_tokens {
            // Prefill ingests a whole prompt at once, so it sweeps a wide set of
            // experts with no reason to revisit any of them. Modelling it means
            // suppressing exactly the structure decode has: low locality, and no
            // carry from the previous token or layer.
            let in_prefill = token < cfg.prefill_tokens;
            let (locality, persistence, layer_coupling) = if in_prefill {
                (cfg.prefill_locality, 0.0, 0.0)
            } else {
                (cfg.locality, cfg.persistence, cfg.layer_coupling)
            };
            let mut prev_layer_cluster: Option<u32> = None;
            for slot in last_cluster.iter_mut() {
                let unit = |v: u64| (v >> 11) as f64 / (1u64 << 53) as f64;
                let clustered = unit(next()) < locality;
                // Three independent sources of structure, resolved in order of
                // strength: carry the previous layer's cluster (cross-layer
                // agreement), else the previous token's (temporal persistence),
                // else draw fresh.
                let couple = prev_layer_cluster.is_some() && unit(next()) < layer_coupling;
                let keep = have_last && unit(next()) < persistence;
                let c = if couple {
                    prev_layer_cluster.expect("checked above")
                } else if keep {
                    *slot
                } else {
                    (next() % u64::from(clusters)) as u32
                };
                *slot = c;
                prev_layer_cluster = Some(c);

                let mut chosen: Vec<u32> = Vec::with_capacity(top_k);
                let mut guard = 0u32;
                while chosen.len() < top_k && guard < 10_000 {
                    guard += 1;
                    let e = if clustered {
                        c * per_cluster + (next() % u64::from(per_cluster)) as u32
                    } else {
                        (next() % u64::from(n_experts)) as u32
                    };
                    let e = e.min(n_experts - 1);
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                while chosen.len() < top_k {
                    let e = (next() % u64::from(n_experts)) as u32;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                selections.extend_from_slice(&chosen);
            }
            have_last = true;
        }
        Trace {
            n_layers: cfg.n_layers,
            n_experts,
            top_k: cfg.top_k,
            selections,
            prefill_tokens: cfg.prefill_tokens,
        }
    }

    /// Generate a synthetic trace with planted cluster structure.
    ///
    /// Useful for tests and for demonstrating the pipeline without a
    /// multi-gigabyte checkpoint. `locality` in `0.0..=1.0` sets how often a
    /// token draws its experts from a single cluster rather than uniformly;
    /// 0.0 approximates random routing, for which no layout can help.
    /// As [`Trace::synthetic_cfg`] with zero persistence.
    ///
    /// Retained because plenty of tests only care about co-activation structure.
    /// Do not use it to evaluate temporal prediction: with no persistence there
    /// is nothing across tokens to predict.
    #[allow(clippy::too_many_arguments)]
    pub fn synthetic(
        n_layers: u32,
        n_experts: u32,
        top_k: u32,
        n_tokens: usize,
        n_clusters: u32,
        locality: f64,
        seed: u64,
    ) -> Self {
        Trace::synthetic_cfg(&SynthConfig {
            n_layers,
            n_experts,
            top_k,
            tokens: n_tokens,
            clusters: n_clusters,
            locality,
            persistence: 0.0,
            layer_coupling: 0.0,
            prefill_tokens: 0,
            prefill_locality: 0.05,
            seed,
        })
    }

    /// Mean number of *distinct* experts selected across windows of `window`
    /// consecutive tokens at one layer.
    ///
    /// This is the quantity speculative decoding turns into bandwidth. Verifying
    /// `window` drafted tokens in a single pass reads the *union* of the experts
    /// those tokens need, not the sum — and because adjacent tokens reuse
    /// experts, the union grows sublinearly. Dividing it by the tokens actually
    /// accepted gives bytes per accepted token, which is the figure that matters
    /// on a bandwidth-bound machine.
    ///
    /// Returns `top_k` for `window <= 1`, and never exceeds `n_experts`.
    pub fn mean_window_union(&self, layer: u32, window: usize) -> f64 {
        let n = self.n_tokens();
        if n == 0 {
            return 0.0;
        }
        let window = window.max(1);
        if window == 1 {
            return f64::from(self.top_k);
        }
        let mut seen = vec![u32::MAX; self.n_experts as usize];
        let mut total = 0u64;
        let mut windows = 0u64;
        // Non-overlapping windows: a speculative pass consumes its whole block
        // before the next one starts, so overlapping windows would double-count
        // reuse that never happens.
        let mut start = 0usize;
        while start + window <= n {
            let stamp = windows as u32;
            let mut distinct = 0u64;
            for tok in start..start + window {
                for &e in self.selection(tok, layer) {
                    if seen[e as usize] != stamp {
                        seen[e as usize] = stamp;
                        distinct += 1;
                    }
                }
            }
            total += distinct;
            windows += 1;
            start += window;
        }
        if windows == 0 {
            return f64::from(self.top_k);
        }
        (total as f64 / windows as f64).min(f64::from(self.n_experts))
    }

    /// Experts never selected at `layer` across the whole trace.
    ///
    /// Strong workload-specific evidence for pruning: bytes streamed for experts
    /// this workload never asks for. Weaker evidence than a saliency measure,
    /// though, and the caveat matters — see the outlier-expert hazard.
    pub fn never_selected(&self, layer: u32) -> Vec<u32> {
        let mut hit = vec![false; self.n_experts as usize];
        for tok in 0..self.n_tokens() {
            for &e in self.selection(tok, layer) {
                hit[e as usize] = true;
            }
        }
        hit.iter()
            .enumerate()
            .filter(|(_, h)| !**h)
            .map(|(e, _)| e as u32)
            .collect()
    }

    /// Apply a pseudorandom relabelling to expert ids.
    ///
    /// [`Trace::synthetic`] plants its clusters as contiguous id ranges, which
    /// means the identity order is *already* optimal — a checkpoint that
    /// flattering does not exist. Real expert numbering carries no locality:
    /// experts that fire together have unrelated indices. Scattering the labels
    /// reproduces that, so a measured layout gain is a real one.
    ///
    /// The relabelling is a bijection, so co-activation structure is preserved
    /// exactly; only the names change.
    pub fn scatter_labels(&mut self, seed: u64) {
        let n = self.n_experts as usize;
        if n < 2 {
            return;
        }
        let mut map: Vec<u32> = (0..n as u32).collect();
        let mut x = seed | 1;
        // Fisher-Yates with a xorshift source, so the result is reproducible.
        for i in (1..n).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let j = (x % (i as u64 + 1)) as usize;
            map.swap(i, j);
        }
        for e in self.selections.iter_mut() {
            *e = map[*e as usize];
        }
    }
}

/// Shape and structure of a synthetic trace.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// MoE layers.
    pub n_layers: u32,
    /// Experts per layer.
    pub n_experts: u32,
    /// Experts selected per token per layer.
    pub top_k: u32,
    /// Tokens to generate.
    pub tokens: usize,
    /// Planted co-activation clusters.
    pub clusters: u32,
    /// Probability a token draws its experts from a single cluster rather than
    /// uniformly. Controls *within-token* co-activation.
    pub locality: f64,
    /// Probability a token reuses the previous token's cluster for a layer.
    ///
    /// Controls *across-token* structure, which is a different thing from
    /// `locality` and easy to conflate. A trace with high locality but zero
    /// persistence has strong co-activation and no temporal correlation at all —
    /// nothing a lookahead predictor can use, and the reason this parameter
    /// exists. Real routing shows adjacent-token overlap around twice a random
    /// baseline, which corresponds to a persistence well above zero.
    pub persistence: f64,
    /// Probability a layer reuses the *previous layer's* cluster within the same
    /// token.
    ///
    /// A third, independent axis. Real MoE stacks show strong cross-layer
    /// agreement — consecutive blocks route to the same expert ids a large
    /// fraction of the time — which is what makes it possible to predict layer
    /// `n` from layer `n-1` and thus to prefetch before the router has run. With
    /// this at zero, each layer routes independently and no cross-layer
    /// predictor can beat guessing the hottest experts.
    pub layer_coupling: f64,
    /// Tokens of prefill to emit *before* the `tokens` decode steps.
    ///
    /// Zero reproduces the undifferentiated traces this generator produced
    /// before phases existed. A non-zero burst is what distinguishes a
    /// segmented cache policy from a recency-only one: without it, SLRU and LRU
    /// are the same measurement.
    pub prefill_tokens: usize,
    /// Locality during the prefill burst.
    ///
    /// Deliberately much lower than [`SynthConfig::locality`]. Prefill's defining
    /// property for a cache is *width* — it touches many experts once each —
    /// and locality is the knob that controls width.
    pub prefill_locality: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig {
            n_layers: 2,
            n_experts: 32,
            top_k: 4,
            tokens: 4000,
            clusters: 4,
            locality: 0.85,
            persistence: 0.6,
            layer_coupling: 0.45,
            prefill_tokens: 0,
            prefill_locality: 0.05,
            seed: 0xC0FFEE,
        }
    }
}

/// Per-layer access frequency and pairwise co-activation counts.
#[derive(Debug, Clone)]
pub struct CoActivation {
    /// Experts covered.
    pub n_experts: u32,
    /// How many tokens selected each expert.
    pub freq: Vec<u64>,
    /// Full `n x n` symmetric count matrix; `pair[i * n + j]` is how often `i`
    /// and `j` were selected for the same token. The diagonal is zero.
    pub pair: Vec<u32>,
    /// Tokens observed.
    pub tokens: u64,
}

impl CoActivation {
    /// Accumulate statistics for one layer of a trace.
    pub fn from_trace(trace: &Trace, layer: u32) -> Self {
        let n = trace.n_experts as usize;
        let mut freq = vec![0u64; n];
        let mut pair = vec![0u32; n * n];
        let mut tokens = 0u64;
        for edge in trace.layer_edges(layer) {
            tokens += 1;
            for (a_idx, &a) in edge.iter().enumerate() {
                freq[a as usize] += 1;
                for &b in &edge[a_idx + 1..] {
                    if a == b {
                        continue;
                    }
                    pair[a as usize * n + b as usize] =
                        pair[a as usize * n + b as usize].saturating_add(1);
                    pair[b as usize * n + a as usize] =
                        pair[b as usize * n + a as usize].saturating_add(1);
                }
            }
        }
        CoActivation {
            n_experts: trace.n_experts,
            freq,
            pair,
            tokens,
        }
    }

    /// Co-activation count for a pair.
    pub fn co(&self, a: u32, b: u32) -> u32 {
        self.pair[a as usize * self.n_experts as usize + b as usize]
    }

    /// Share of total selections taken by the `top` most-used experts.
    ///
    /// This is the skew that determines how effective a small resident cache can
    /// be. A uniform router returns `top / n_experts`; anything above that is
    /// headroom a cache can exploit.
    pub fn mass_in_top(&self, top: usize) -> f64 {
        let total: u64 = self.freq.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let mut f = self.freq.clone();
        f.sort_unstable_by(|a, b| b.cmp(a));
        let head: u64 = f.iter().take(top).sum();
        head as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_trace_has_the_declared_shape() {
        let t = Trace::synthetic(4, 32, 8, 500, 4, 0.9, 7);
        t.validate().expect("valid");
        assert_eq!(t.n_tokens(), 500);
        assert_eq!(t.selection(10, 2).len(), 8);
    }

    #[test]
    fn synthetic_selections_are_distinct_within_a_token() {
        let t = Trace::synthetic(2, 16, 8, 200, 2, 0.8, 3);
        for tok in 0..t.n_tokens() {
            for l in 0..t.n_layers {
                let s = t.selection(tok, l);
                let mut v = s.to_vec();
                v.sort_unstable();
                v.dedup();
                assert_eq!(v.len(), s.len(), "token {tok} layer {l} repeated an expert");
            }
        }
    }

    #[test]
    fn locality_raises_co_activation_within_clusters() {
        let n = 32;
        let clustered = Trace::synthetic(1, n, 4, 4000, 4, 0.95, 11);
        let random = Trace::synthetic(1, n, 4, 4000, 4, 0.0, 11);
        let ca = CoActivation::from_trace(&clustered, 0);
        let ra = CoActivation::from_trace(&random, 0);
        // Experts 0 and 1 share a cluster (32 experts / 4 clusters = 8 wide).
        assert!(
            ca.co(0, 1) > ra.co(0, 1) * 2,
            "clustered co-activation {} should far exceed random {}",
            ca.co(0, 1),
            ra.co(0, 1)
        );
    }

    #[test]
    fn scatter_preserves_co_activation_structure() {
        let base = Trace::synthetic(1, 32, 4, 2000, 4, 0.95, 41);
        let mut scattered = base.clone();
        scattered.scatter_labels(99);
        scattered.validate().expect("still valid");

        let a = CoActivation::from_trace(&base, 0);
        let b = CoActivation::from_trace(&scattered, 0);
        // The multiset of pairwise counts is invariant under relabelling.
        let mut va: Vec<u32> = (0..32u32)
            .flat_map(|i| (0..32u32).map(move |j| (i, j)))
            .map(|(i, j)| a.co(i, j))
            .collect();
        let mut vb: Vec<u32> = (0..32u32)
            .flat_map(|i| (0..32u32).map(move |j| (i, j)))
            .map(|(i, j)| b.co(i, j))
            .collect();
        va.sort_unstable();
        vb.sort_unstable();
        assert_eq!(
            va, vb,
            "relabelling must not change the co-activation multiset"
        );
    }

    #[test]
    fn scatter_breaks_index_adjacency() {
        // Contiguous clusters make neighbouring ids co-fire; scattering must not.
        let base = Trace::synthetic(1, 32, 4, 4000, 4, 1.0, 7);
        let mut sc = base.clone();
        sc.scatter_labels(5);
        let a = CoActivation::from_trace(&base, 0);
        let b = CoActivation::from_trace(&sc, 0);
        let adj_a: u64 = (0..31u32).map(|i| u64::from(a.co(i, i + 1))).sum();
        let adj_b: u64 = (0..31u32).map(|i| u64::from(b.co(i, i + 1))).sum();
        assert!(
            adj_b < adj_a,
            "scattered adjacency {adj_b} should fall below contiguous {adj_a}"
        );
    }

    #[test]
    fn window_union_grows_sublinearly_with_reuse() {
        // The mechanism speculative decoding exploits: with temporal reuse, the
        // union over a block of tokens is far smaller than the sum, so one
        // verification pass reads less per token than one-at-a-time decoding.
        let cfg = |persistence: f64| SynthConfig {
            n_layers: 1,
            n_experts: 64,
            top_k: 8,
            tokens: 4000,
            clusters: 8,
            locality: 0.9,
            persistence,
            layer_coupling: 0.0,
            seed: 19,
            prefill_tokens: 0,
            prefill_locality: 0.05,
        };
        let reuse = Trace::synthetic_cfg(&cfg(0.9));
        let indep = Trace::synthetic_cfg(&cfg(0.0));

        for w in [2usize, 4, 8] {
            let u = reuse.mean_window_union(0, w);
            let sum = 8.0 * w as f64;
            assert!(u < sum, "window {w}: union {u} should beat the sum {sum}");
            // Reuse must beat independence at the same window.
            let ui = indep.mean_window_union(0, w);
            assert!(u < ui, "window {w}: reuse {u} should beat independent {ui}");
        }
        assert_eq!(reuse.mean_window_union(0, 1), 8.0);
        // Never more experts than exist.
        assert!(reuse.mean_window_union(0, 1000) <= 64.0);
    }

    #[test]
    fn never_selected_finds_unused_experts() {
        // Only ids 0..8 are ever used, so the rest are prunable for this workload.
        let t = Trace {
            n_layers: 1,
            n_experts: 32,
            top_k: 4,
            selections: (0..400u32).map(|i| i % 8).collect(),
            prefill_tokens: 0,
        };
        t.validate().unwrap();
        let unused = t.never_selected(0);
        assert_eq!(unused.len(), 24);
        assert!(unused.iter().all(|e| *e >= 8));

        // A trace touching everything leaves nothing prunable.
        let full = Trace::synthetic(1, 16, 8, 2000, 1, 0.0, 3);
        assert!(full.never_selected(0).is_empty());
    }

    #[test]
    fn persistence_creates_adjacent_token_overlap() {
        // Without persistence, consecutive tokens share experts only by chance.
        // This is the property lookahead prediction depends on, and it is
        // independent of within-token co-activation.
        let overlap = |persistence: f64| -> f64 {
            let t = Trace::synthetic_cfg(&SynthConfig {
                n_layers: 1,
                n_experts: 64,
                top_k: 6,
                tokens: 4000,
                clusters: 8,
                locality: 0.9,
                persistence,
                layer_coupling: 0.0,
                seed: 31,
                prefill_tokens: 0,
                prefill_locality: 0.05,
            });
            let mut shared = 0u64;
            for tok in 1..t.n_tokens() {
                let a = t.selection(tok - 1, 0);
                let b = t.selection(tok, 0);
                shared += b.iter().filter(|e| a.contains(e)).count() as u64;
            }
            shared as f64 / ((t.n_tokens() - 1) * 6) as f64
        };
        let none = overlap(0.0);
        let high = overlap(0.9);
        assert!(
            high > none * 1.8,
            "persistence 0.9 gave adjacent overlap {high:.3} vs {none:.3} at zero; \
             lookahead prediction has nothing to learn without this"
        );
    }

    #[test]
    fn layer_coupling_creates_cross_layer_agreement() {
        // The property that makes it possible to predict layer n from layer n-1,
        // and therefore to have reads in flight before the router has run.
        let agreement = |coupling: f64| -> f64 {
            let t = Trace::synthetic_cfg(&SynthConfig {
                n_layers: 6,
                n_experts: 64,
                top_k: 6,
                tokens: 3000,
                clusters: 8,
                locality: 0.9,
                persistence: 0.0,
                layer_coupling: coupling,
                seed: 77,
                prefill_tokens: 0,
                prefill_locality: 0.05,
            });
            let mut shared = 0u64;
            let mut n = 0u64;
            for tok in 0..t.n_tokens() {
                for l in 1..t.n_layers {
                    let a = t.selection(tok, l - 1);
                    let b = t.selection(tok, l);
                    shared += b.iter().filter(|e| a.contains(e)).count() as u64;
                    n += b.len() as u64;
                }
            }
            shared as f64 / n as f64
        };
        let none = agreement(0.0);
        let high = agreement(0.9);
        assert!(
            high > none * 1.8,
            "coupling 0.9 gave cross-layer agreement {high:.3} vs {none:.3} at zero"
        );
    }

    #[test]
    fn co_activation_matrix_is_symmetric_with_zero_diagonal() {
        let t = Trace::synthetic(1, 12, 4, 300, 3, 0.7, 5);
        let ca = CoActivation::from_trace(&t, 0);
        for i in 0..12u32 {
            assert_eq!(ca.co(i, i), 0);
            for j in 0..12u32 {
                assert_eq!(ca.co(i, j), ca.co(j, i));
            }
        }
    }

    #[test]
    fn skew_exceeds_uniform_when_routing_is_clustered() {
        // Bias frequency by drawing from few clusters.
        let t = Trace::synthetic(1, 64, 4, 4000, 2, 1.0, 9);
        let ca = CoActivation::from_trace(&t, 0);
        // Top 16 of 64 would be 25% under uniform routing.
        assert!(ca.mass_in_top(16) > 0.25);
    }

    #[test]
    fn text_round_trips_through_binary() {
        let text = "\
# token layer experts
0 0 1 2 3
0 1 4 5 6
1 0 1 2 7
1 1 4 5 6
";
        let t = Trace::from_text(text).expect("parses");
        assert_eq!(t.n_layers, 2);
        assert_eq!(t.top_k, 3);
        assert_eq!(t.n_tokens(), 2);
        assert_eq!(t.selection(1, 0), &[1, 2, 7]);

        let dir = std::env::temp_dir().join("pmx-trace-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.pmxtrace");
        t.write(&p).unwrap();
        let back = Trace::read(&p).unwrap();
        assert_eq!(back.selections, t.selections);
        assert_eq!(back.top_k, t.top_k);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn text_rejects_missing_layers() {
        // Token 1 never records layer 1.
        let text = "0 0 1 2\n0 1 3 4\n1 0 1 2\n";
        assert!(Trace::from_text(text).is_err());
    }

    #[test]
    fn out_of_range_expert_is_rejected() {
        let t = Trace {
            n_layers: 1,
            n_experts: 4,
            top_k: 2,
            selections: vec![0, 9],
            prefill_tokens: 0,
        };
        assert!(matches!(
            t.validate(),
            Err(TraceError::ExpertOutOfRange {
                id: 9,
                n_experts: 4
            })
        ));
    }

    #[test]
    fn ragged_body_is_rejected() {
        let t = Trace {
            n_layers: 2,
            n_experts: 4,
            top_k: 2,
            selections: vec![0, 1, 2],
            prefill_tokens: 0,
        };
        assert!(matches!(t.validate(), Err(TraceError::Ragged { .. })));
    }
    #[test]
    fn prefill_covers_more_experts_than_decode() {
        // The property the whole phase split exists to produce. If prefill were
        // not wider than decode there would be nothing for a segmented policy
        // to defend against, and the SLRU measurement would be meaningless.
        let t = Trace::synthetic_cfg(&SynthConfig {
            n_layers: 1,
            n_experts: 64,
            top_k: 4,
            tokens: 200,
            prefill_tokens: 200,
            clusters: 8,
            locality: 0.9,
            persistence: 0.85,
            ..Default::default()
        });
        // Total coverage is the wrong measure and saturates: over 800 draws both
        // phases eventually touch all 64 experts. What a cache actually feels is
        // the *working-set width* -- distinct experts within a short window --
        // so measure that, over non-overlapping windows.
        let width = |sel: &[u32], window: usize| -> f64 {
            let per = window * t.per_token();
            let windows = sel.len() / per;
            assert!(windows > 0, "window longer than the phase");
            let mut total = 0usize;
            for w in 0..windows {
                let mut seen = vec![false; t.n_experts as usize];
                for &e in &sel[w * per..(w + 1) * per] {
                    seen[e as usize] = true;
                }
                total += seen.iter().filter(|s| **s).count();
            }
            total as f64 / windows as f64
        };
        let pre = width(t.prefill_selections(), 8);
        let dec = width(t.decode_selections(), 8);
        assert_eq!(t.prefill_tokens, 200);
        assert_eq!(t.decode_tokens(), 200);
        assert!(
            pre > dec * 1.5,
            "prefill width {pre:.1} should clearly exceed decode width {dec:.1}"
        );
    }

    #[test]
    fn phases_partition_the_selections_exactly() {
        let t = Trace::synthetic_cfg(&SynthConfig {
            n_layers: 3,
            n_experts: 32,
            top_k: 4,
            tokens: 50,
            prefill_tokens: 20,
            ..Default::default()
        });
        assert_eq!(
            t.prefill_selections().len() + t.decode_selections().len(),
            t.selections.len()
        );
        assert_eq!(t.n_tokens(), 70);
        assert_eq!(t.prefill_selections().len(), 20 * t.per_token());
    }

    #[test]
    fn the_phase_boundary_survives_a_round_trip() {
        let dir = std::env::temp_dir().join("pmx-trace-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("phased.pmxtrace");
        let t = Trace::synthetic_cfg(&SynthConfig {
            n_layers: 2,
            n_experts: 16,
            top_k: 2,
            tokens: 30,
            prefill_tokens: 11,
            ..Default::default()
        });
        t.write(&p).unwrap();
        let back = Trace::read(&p).unwrap();
        assert_eq!(back.prefill_tokens, 11);
        assert_eq!(back.selections, t.selections);
    }

    #[test]
    fn version_1_traces_still_read_as_all_decode() {
        // Traces written before phases existed must keep working, and the honest
        // reading of one is that it is undifferentiated -- not that it is all
        // prefill. Getting this backwards would silently flatter SLRU.
        let dir = std::env::temp_dir().join("pmx-trace-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v1.pmxtrace");
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION_V1.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        buf.extend_from_slice(&8u32.to_le_bytes()); // n_experts
        buf.extend_from_slice(&2u32.to_le_bytes()); // top_k
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_tokens
        for v in [0u32, 1, 2, 3, 4, 5] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&p, &buf).unwrap();
        let t = Trace::read(&p).unwrap();
        assert_eq!(t.prefill_tokens, 0);
        assert_eq!(t.n_tokens(), 3);
        assert_eq!(t.decode_selections().len(), 6);
    }

    #[test]
    fn a_prefill_boundary_past_the_end_is_rejected() {
        let t = Trace {
            n_layers: 1,
            n_experts: 8,
            top_k: 2,
            selections: vec![0, 1, 2, 3],
            prefill_tokens: 99,
        };
        assert!(matches!(t.validate(), Err(TraceError::BadHeader(_))));
    }
}
