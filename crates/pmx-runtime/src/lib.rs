//! The streaming expert-fetch runtime.
//!
//! # What this measures, and what it does not
//!
//! This drives the weight-movement half of MoE decoding: for each token and
//! layer, predict the experts about to be needed, issue their reads with enough
//! in flight to keep the device busy, serve what the cache already holds, and
//! account for the rest. It replays a routing trace rather than running a
//! transformer, so it reports **expert-fetch throughput**, not tokens per second
//! of generated text.
//!
//! That is a real limitation and worth stating plainly. But it is also the
//! quantity that decides whether disk-resident MoE inference is viable at all:
//! on a memory-constrained machine, decoding is bound by moving expert weights,
//! not by arithmetic. Existing engines collapse here — one CPU-only report puts
//! a 25 GB box at 0.05-0.1 tok/s — and a harness that isolates the bottleneck is
//! how you find out whether a change helps before building a whole engine around
//! it.
//!
//! # The three levers, in one place
//!
//! * **Prefetch** ([`pmx_predict`]) buys queue depth, which the measured
//!   bandwidth surface shows to be worth 6-8x. Nothing else in the system can
//!   claim that much.
//! * **Cache** ([`pmx_cache`]) avoids the read entirely, and being cost-aware
//!   avoids the *expensive* reads preferentially.
//! * **Precision** ([`pmx_store`]) shrinks the read. Independent of the other
//!   two, and the only one that shows up on a flat bandwidth surface.
//!
//! [`replay`] reports each separately so their contributions can be told apart.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pmx_cache::{CacheStats, ExpertCache, ExpertId, Policy};
use pmx_predict::{Predictor, RoutingModel};
use pmx_probe::Surface;
use pmx_store::Store;
use pmx_trace::Trace;

/// How to run a replay.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Bytes of RAM available for routed experts.
    pub cache_bytes: u64,
    /// Cache replacement policy.
    pub policy: Policy,
    /// Which lookahead predictor to use, or `None` to fetch only on demand.
    pub predictor: Option<Predictor>,
    /// Experts the predictor may name per layer. Above `top_k` this trades
    /// bandwidth for recall.
    pub prefetch_budget: usize,
    /// Reads the fetcher keeps in flight. Sets which column of the measured
    /// bandwidth surface applies.
    pub queue_depth: usize,
    /// Tokens used to fit the predictor before measurement begins.
    pub fit_tokens: usize,
    /// Measured device bandwidth surface. Without one, times are not reported.
    pub surface: Surface,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            cache_bytes: 1 << 30,
            policy: Policy::Gdsf,
            predictor: Some(Predictor::StickyMarkov),
            prefetch_budget: 16,
            queue_depth: 8,
            fit_tokens: 0,
            surface: Surface::default(),
        }
    }
}

/// What a replay measured.
#[derive(Debug, Clone)]
pub struct ReplayReport {
    /// Tokens replayed, excluding the predictor's fitting prefix.
    pub tokens: u64,
    /// Cache behaviour over the measured window.
    pub cache: CacheStats,
    /// Bytes of expert weight moved from storage per token.
    pub bytes_per_token: f64,
    /// Read requests issued per token, after coalescing an expert's slices.
    pub requests_per_token: f64,
    /// Predicted seconds of storage read per token.
    pub seconds_per_token: f64,
    /// Experts prefetched that the router then did not select.
    pub wasted_prefetches: u64,
    /// Experts the router selected that a prefetch had already secured.
    pub useful_prefetches: u64,
    /// Whether timings are calibrated by a measured surface.
    pub calibrated: bool,
    /// Effective read bandwidth implied by the above.
    pub effective_bps: f64,
}

impl ReplayReport {
    /// Expert-fetch throughput, in tokens per second.
    ///
    /// This is the rate at which weights can be moved, and therefore a ceiling
    /// on decode rate — not a measured generation speed.
    pub fn fetch_limited_tokens_per_sec(&self) -> f64 {
        if self.seconds_per_token <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.seconds_per_token
        }
    }

    /// Share of prefetched experts that were actually used.
    pub fn prefetch_precision(&self) -> f64 {
        let total = self.useful_prefetches + self.wasted_prefetches;
        if total == 0 {
            0.0
        } else {
            self.useful_prefetches as f64 / total as f64
        }
    }
}

/// Bandwidth for a request of `bytes` at `queue_depth`, from the surface.
fn bandwidth(surface: &Surface, bytes: u64, queue_depth: usize) -> Option<f64> {
    surface.bandwidth_at(bytes.max(1), queue_depth)
}

/// Replay `trace` against `store`, measuring the fetch path.
///
/// The trace's layer indices are taken as MoE layer indices, and its expert ids
/// as store slots — so if the store was written from a repacked checkpoint, the
/// trace must be expressed in the same permuted space.
pub fn replay(store: &Store, trace: &Trace, cfg: &RuntimeConfig) -> ReplayReport {
    let n_tokens = trace.n_tokens();
    let fit = cfg.fit_tokens.min(n_tokens);
    let model = cfg.predictor.map(|_| RoutingModel::fit(trace, fit.max(1)));

    // Per-expert byte cost, from the store. Experts absent from the store are
    // treated as zero-cost, which keeps a partial store usable for experiments.
    let bytes_of = |layer: u32, slot: u32| -> u64 { store.expert_bytes(layer, slot) };

    let mut cache = ExpertCache::new(cfg.cache_bytes, cfg.policy);
    let mut bytes_total = 0u64;
    let mut requests_total = 0u64;
    let mut seconds_total = 0.0f64;
    let mut wasted = 0u64;
    let mut useful = 0u64;
    let mut measured_tokens = 0u64;
    let mut pred_buf: Vec<u32> = Vec::new();
    // Experts a prefetch has brought in but the router has not yet asked for.
    let mut in_flight: Vec<ExpertId> = Vec::new();

    let calibrated = !cfg.surface.cells.is_empty();

    for t in fit..n_tokens {
        measured_tokens += 1;
        for l in 0..trace.n_layers {
            // --- prefetch, using only what a real engine would already know ---
            if let (Some(m), Some(which)) = (model.as_ref(), cfg.predictor) {
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
                m.predict(
                    which,
                    l,
                    prev_layer,
                    prev_token,
                    cfg.prefetch_budget,
                    &mut pred_buf,
                );

                for &e in &pred_buf {
                    let id = (l, e);
                    if cache.contains(id) {
                        continue;
                    }
                    let b = bytes_of(l, e);
                    if b == 0 {
                        continue;
                    }
                    // A prefetch is a real read: charge for it whether or not the
                    // router turns out to want it. Counting only useful
                    // prefetches would make any predictor look free.
                    let cost = charge(&cfg.surface, b, cfg.queue_depth, calibrated);
                    bytes_total += b;
                    requests_total += 1;
                    seconds_total += cost;
                    // Install without recording a miss: the accounting above
                    // already covers the fetch.
                    cache.prefetch(id, b, cost);
                    in_flight.push(id);
                }
            }

            // --- the router's real choice ---
            let truth = trace.selection(t, l);
            for &e in truth {
                let id = (l, e);
                let b = bytes_of(l, e);
                if b == 0 {
                    continue;
                }
                let was_prefetched = in_flight.contains(&id);
                let cost = charge(&cfg.surface, b, cfg.queue_depth, calibrated);
                let hit = cache.access(id, b, cost);
                if hit {
                    if was_prefetched {
                        useful += 1;
                    }
                } else {
                    // A genuine on-demand fetch, at queue depth 1: nothing told
                    // us to ask for it earlier. This is the cost prediction
                    // exists to avoid, and pricing it at QD1 is the point.
                    let demand_cost = charge(&cfg.surface, b, 1, calibrated);
                    bytes_total += b;
                    requests_total += 1;
                    seconds_total += demand_cost;
                }
            }
            // Anything prefetched for this layer that the router did not want.
            for id in in_flight.drain(..) {
                if !truth.contains(&id.1) {
                    wasted += 1;
                }
            }
        }
    }

    let tok = measured_tokens.max(1) as f64;
    let bytes_per_token = bytes_total as f64 / tok;
    let effective_bps = if seconds_total > 0.0 {
        bytes_total as f64 / seconds_total
    } else {
        0.0
    };
    ReplayReport {
        tokens: measured_tokens,
        cache: cache.stats(),
        bytes_per_token,
        requests_per_token: requests_total as f64 / tok,
        seconds_per_token: seconds_total / tok,
        wasted_prefetches: wasted,
        useful_prefetches: useful,
        calibrated,
        effective_bps,
    }
}

/// Seconds to move `bytes` at `queue_depth`.
///
/// With no measured surface, returns a request count so relative comparisons
/// still work. Callers must not present that as a time — [`ReplayReport`]
/// carries `calibrated` for exactly this reason.
fn charge(surface: &Surface, bytes: u64, queue_depth: usize, calibrated: bool) -> f64 {
    if !calibrated {
        return 1.0;
    }
    match bandwidth(surface, bytes, queue_depth) {
        Some(bps) if bps > 0.0 => bytes as f64 / bps,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmx_kernels::{PmxType, GROUP};
    use pmx_probe::Cell;
    use pmx_store::{Kind, StoreWriter};
    use pmx_trace::SynthConfig;
    use std::path::PathBuf;

    fn surface() -> Surface {
        let raw: &[(u64, usize, f64)] = &[
            (4 << 10, 1, 0.02),
            (4 << 10, 4, 0.06),
            (4 << 10, 8, 0.09),
            (4 << 10, 16, 0.13),
            (16 << 10, 1, 0.06),
            (16 << 10, 4, 0.16),
            (16 << 10, 8, 0.25),
            (16 << 10, 16, 0.48),
            (64 << 10, 1, 0.15),
            (64 << 10, 4, 0.38),
            (64 << 10, 8, 0.61),
            (64 << 10, 16, 0.91),
            (256 << 10, 1, 0.35),
            (256 << 10, 4, 1.24),
            (256 << 10, 8, 1.82),
            (256 << 10, 16, 2.16),
            (1 << 20, 1, 0.91),
            (1 << 20, 8, 2.51),
            (2 << 20, 8, 2.54),
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

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join("pmx-runtime-tests");
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// A store with `experts` experts per layer, each slice `w` weights.
    fn store_at(path: &PathBuf, layers: u32, experts: u32, w: usize, ty: PmxType) -> Store {
        let mut sw = StoreWriter::new(4096, 8);
        let vals: Vec<f32> = (0..w).map(|i| (i % 17) as f32 * 0.01).collect();
        for l in 0..layers {
            for e in 0..experts {
                for k in Kind::ALL {
                    sw.add(l, e, k, ty, &vals).unwrap();
                }
            }
        }
        sw.finish(path).unwrap();
        Store::open(path).unwrap()
    }

    fn trace() -> Trace {
        let mut t = Trace::synthetic_cfg(&SynthConfig {
            n_layers: 4,
            n_experts: 32,
            top_k: 4,
            tokens: 1200,
            clusters: 8,
            locality: 0.9,
            persistence: 0.7,
            layer_coupling: 0.45,
            seed: 3,
        });
        t.scatter_labels(0xBEEF);
        t
    }

    #[test]
    fn a_replay_accounts_for_every_token() {
        let p = tmp("basic.pmxstore");
        let s = store_at(&p, 4, 32, GROUP * 2, PmxType::Q4);
        let t = trace();
        let cfg = RuntimeConfig {
            cache_bytes: 64 * 1024,
            fit_tokens: 400,
            surface: surface(),
            ..Default::default()
        };
        let r = replay(&s, &t, &cfg);
        assert_eq!(r.tokens, (t.n_tokens() - 400) as u64);
        assert!(r.calibrated);
        assert!(r.bytes_per_token > 0.0);
        assert!(r.seconds_per_token > 0.0);
        assert!(r.fetch_limited_tokens_per_sec().is_finite());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_bigger_cache_moves_fewer_bytes() {
        let p = tmp("cachesize.pmxstore");
        let s = store_at(&p, 4, 32, GROUP * 2, PmxType::Q4);
        let t = trace();
        let run = |bytes: u64| {
            replay(
                &s,
                &t,
                &RuntimeConfig {
                    cache_bytes: bytes,
                    fit_tokens: 400,
                    predictor: None,
                    surface: surface(),
                    ..Default::default()
                },
            )
        };
        let small = run(16 * 1024);
        let large = run(4 * 1024 * 1024);
        assert!(
            large.bytes_per_token < small.bytes_per_token,
            "a larger cache should move fewer bytes: {:.0} vs {:.0}",
            large.bytes_per_token,
            small.bytes_per_token
        );
        assert!(large.cache.hit_rate() > small.cache.hit_rate());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn lower_precision_moves_fewer_bytes_for_the_same_trace() {
        // The lever that works even on a flat bandwidth surface.
        let p4 = tmp("prec4.pmxstore");
        let p2 = tmp("prec2.pmxstore");
        let s4 = store_at(&p4, 4, 32, GROUP * 2, PmxType::Q4);
        let s2 = store_at(&p2, 4, 32, GROUP * 2, PmxType::Q2);
        let t = trace();
        let cfg = RuntimeConfig {
            cache_bytes: 32 * 1024,
            fit_tokens: 400,
            predictor: None,
            surface: surface(),
            ..Default::default()
        };
        let r4 = replay(&s4, &t, &cfg);
        let r2 = replay(&s2, &t, &cfg);
        assert!(
            r2.bytes_per_token < r4.bytes_per_token,
            "Q2 moved {:.0} bytes/token vs Q4 {:.0}",
            r2.bytes_per_token,
            r4.bytes_per_token
        );
        let _ = std::fs::remove_file(&p4);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn prefetching_trades_bytes_for_time() {
        // Prefetch fetches more (some predictions go unused) but at queue depth,
        // where the device is far faster. On a surface with a real queue-depth
        // gradient the time must come down even though the bytes go up.
        let p = tmp("prefetch.pmxstore");
        let s = store_at(&p, 4, 32, GROUP * 2, PmxType::Q4);
        let t = trace();
        let base = RuntimeConfig {
            cache_bytes: 24 * 1024,
            fit_tokens: 400,
            queue_depth: 16,
            surface: surface(),
            ..Default::default()
        };
        let demand = replay(
            &s,
            &t,
            &RuntimeConfig {
                predictor: None,
                ..base.clone()
            },
        );
        let pre = replay(
            &s,
            &t,
            &RuntimeConfig {
                predictor: Some(Predictor::StickyMarkov),
                prefetch_budget: 8,
                ..base.clone()
            },
        );
        assert!(
            pre.bytes_per_token > demand.bytes_per_token,
            "prefetch should move more bytes: {:.0} vs {:.0}",
            pre.bytes_per_token,
            demand.bytes_per_token
        );
        assert!(
            pre.seconds_per_token < demand.seconds_per_token,
            "prefetch should still be faster: {:.6}s vs {:.6}s per token",
            pre.seconds_per_token,
            demand.seconds_per_token
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn prefetch_precision_is_reported_and_bounded() {
        let p = tmp("precision.pmxstore");
        let s = store_at(&p, 4, 32, GROUP * 2, PmxType::Q4);
        let t = trace();
        let r = replay(
            &s,
            &t,
            &RuntimeConfig {
                cache_bytes: 24 * 1024,
                fit_tokens: 400,
                predictor: Some(Predictor::StickyMarkov),
                prefetch_budget: 12,
                surface: surface(),
                ..Default::default()
            },
        );
        let prec = r.prefetch_precision();
        assert!((0.0..=1.0).contains(&prec), "precision {prec} out of range");
        assert!(
            r.useful_prefetches > 0,
            "no prefetch was ever useful; the predictor is not wired up"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_uncalibrated_run_says_so() {
        let p = tmp("uncal.pmxstore");
        let s = store_at(&p, 2, 16, GROUP, PmxType::Q4);
        let t = trace();
        let r = replay(
            &s,
            &t,
            &RuntimeConfig {
                fit_tokens: 400,
                surface: Surface::default(),
                ..Default::default()
            },
        );
        assert!(
            !r.calibrated,
            "a run with no surface must not claim calibration"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_empty_store_does_not_panic() {
        let p = tmp("empty.pmxstore");
        StoreWriter::new(4096, 8).finish(&p).unwrap();
        let s = Store::open(&p).unwrap();
        let r = replay(
            &s,
            &trace(),
            &RuntimeConfig {
                fit_tokens: 400,
                ..Default::default()
            },
        );
        assert_eq!(r.bytes_per_token, 0.0);
        let _ = std::fs::remove_file(&p);
    }
}
