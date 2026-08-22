//! The expert residency cache.
//!
//! # Why not LRU
//!
//! Expert-offloading systems almost universally use LRU, justified by the
//! observation that consecutive tokens reuse experts. That justification is
//! sound but incomplete: it explains why *recency* carries signal, not why
//! recency is the only thing worth tracking. Three other quantities matter here:
//!
//! * **Frequency.** MoE routing is strongly skewed. A handful of experts absorb
//!   a disproportionate share of selections, and they should never be evicted
//!   because a long tail of one-off experts marched past them.
//! * **Cost.** Once experts are stored at different precisions — which is the
//!   point of `pmx-plan` — they no longer cost the same to fetch. Evicting a
//!   36 KiB expert and a 9 KiB expert are not equivalent decisions.
//! * **Size.** For the same reason, a large expert must earn its residency
//!   against the several small ones that could occupy the space instead.
//!
//! The classical policy combining all four is **GDSF** (Greedy-Dual Size with
//! Frequency): each entry gets a key `L + frequency * cost / size`, where `L` is
//! an inflation term carried forward from the last eviction. `L` is what keeps
//! the policy from ossifying — without it, an entry that was hot early would
//! keep its high key forever and never be reconsidered.
//!
//! [`Policy`] implements GDSF alongside LRU and LFU so the choice can be
//! measured rather than assumed — and measurement produced a result worth
//! stating plainly.
//!
//! **GDSF only beats LRU when cost per byte actually varies.** Its key contains
//! `cost / size`, so if fetch cost is proportional to size — which it is at a
//! fixed bandwidth, since `cost = bytes / rate` — that ratio is constant, the
//! cost term cancels, and GDSF collapses to LFU. And LFU is *worse* than LRU on
//! skewed routing, because unaged frequency counts let an early-hot entry
//! ossify.
//!
//! So cost-awareness pays exactly where cost per byte differs:
//!
//! * across storage tiers (a disk byte costs ~12x a RAM byte on the development
//!   machine), or
//! * where request size changes efficiency — the measured bandwidth surface
//!   shows a 16 KiB read running at 0.25 GB/s against 1.82 GB/s at 256 KiB, so
//!   a small expert really is several times more expensive per byte.
//!
//! Where neither holds, LRU is the better default and this crate will say so.
//! Pick a policy from a measurement, not from a paper.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashMap;

/// Identifies an expert within a model: `(layer, expert)`.
pub type ExpertId = (u32, u32);

/// Which replacement policy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Evict the least recently used entry.
    Lru,
    /// Evict the least frequently used entry.
    Lfu,
    /// Greedy-Dual Size with Frequency: `L + freq * cost / size`.
    Gdsf,
}

impl Policy {
    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Policy::Lru => "lru",
            Policy::Lfu => "lfu",
            Policy::Gdsf => "gdsf",
        }
    }

    /// Parse a policy name.
    pub fn parse(s: &str) -> Option<Policy> {
        Some(match s {
            "lru" => Policy::Lru,
            "lfu" => Policy::Lfu,
            "gdsf" => Policy::Gdsf,
            _ => return None,
        })
    }
}

/// What one cached expert costs and how it has been used.
#[derive(Debug, Clone, Copy)]
struct Entry {
    bytes: u64,
    /// Seconds to fetch this expert from storage, from the tier it lives on.
    fetch_cost: f64,
    hits: u64,
    last_used: u64,
    /// GDSF key at insertion or last hit.
    key: f64,
}

/// Running counters for a cache run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CacheStats {
    /// Lookups that found the expert resident.
    pub hits: u64,
    /// Lookups that had to fetch.
    pub misses: u64,
    /// Entries evicted.
    pub evictions: u64,
    /// Bytes fetched from storage.
    pub bytes_fetched: u64,
    /// Summed fetch cost, in seconds, of every miss.
    pub fetch_seconds: f64,
}

impl CacheStats {
    /// Share of lookups served from RAM.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// A byte-bounded expert cache.
#[derive(Debug)]
pub struct ExpertCache {
    capacity: u64,
    used: u64,
    policy: Policy,
    entries: HashMap<ExpertId, Entry>,
    /// Pinned experts are never evicted, whatever the policy says.
    pinned: HashMap<ExpertId, ()>,
    clock: u64,
    /// GDSF inflation term.
    inflation: f64,
    stats: CacheStats,
}

impl ExpertCache {
    /// A cache holding at most `capacity` bytes.
    pub fn new(capacity: u64, policy: Policy) -> Self {
        ExpertCache {
            capacity,
            used: 0,
            policy,
            entries: HashMap::new(),
            pinned: HashMap::new(),
            clock: 0,
            inflation: 0.0,
            stats: CacheStats::default(),
        }
    }

    /// Counters accumulated so far.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Bytes currently resident.
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Number of resident experts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `id` is resident.
    pub fn contains(&self, id: ExpertId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Pin an expert so it is never evicted.
    ///
    /// This is how a plan's residency decision is enforced: experts the planner
    /// chose to keep resident are pinned, and the policy manages only whatever
    /// space is left over.
    pub fn pin(&mut self, id: ExpertId, bytes: u64, fetch_cost: f64) -> bool {
        if self.pinned.contains_key(&id) {
            return true;
        }
        if !self.entries.contains_key(&id) {
            if self.used + bytes > self.capacity {
                // Make room, but never by evicting another pinned entry.
                if !self.make_room(bytes) {
                    return false;
                }
            }
            self.insert_raw(id, bytes, fetch_cost);
        }
        self.pinned.insert(id, ());
        true
    }

    /// Whether `id` is pinned.
    pub fn is_pinned(&self, id: ExpertId) -> bool {
        self.pinned.contains_key(&id)
    }

    /// Install an expert brought in by a prefetch.
    ///
    /// Deliberately does *not* touch the hit/miss counters: a prefetch is not a
    /// lookup, and folding it into the hit rate would make speculative fetching
    /// look free. The caller accounts for the bytes and time it spent; this only
    /// records that the expert is now resident.
    ///
    /// Returns `true` if the expert was newly installed.
    pub fn prefetch(&mut self, id: ExpertId, bytes: u64, fetch_cost: f64) -> bool {
        if self.entries.contains_key(&id) {
            return false;
        }
        if bytes > self.capacity {
            return false;
        }
        if self.used + bytes > self.capacity && !self.make_room(bytes) {
            return false;
        }
        self.clock += 1;
        self.insert_raw(id, bytes, fetch_cost);
        true
    }

    /// Release a pin, leaving the entry subject to the policy again.
    pub fn unpin(&mut self, id: ExpertId) -> bool {
        self.pinned.remove(&id).is_some()
    }

    /// Record a lookup for `id`, fetching it if absent.
    ///
    /// Returns `true` on a hit. `bytes` and `fetch_cost` describe what fetching
    /// this expert would cost, and are used both for accounting and by GDSF.
    pub fn access(&mut self, id: ExpertId, bytes: u64, fetch_cost: f64) -> bool {
        self.clock += 1;
        if let Some(e) = self.entries.get_mut(&id) {
            e.hits += 1;
            e.last_used = self.clock;
            e.key = self.inflation + e.hits as f64 * e.fetch_cost / e.bytes.max(1) as f64;
            self.stats.hits += 1;
            return true;
        }
        self.stats.misses += 1;
        self.stats.bytes_fetched += bytes;
        self.stats.fetch_seconds += fetch_cost;

        if bytes > self.capacity {
            // Larger than the whole cache; it can be used but not retained.
            return false;
        }
        if self.used + bytes > self.capacity && !self.make_room(bytes) {
            // Everything resident is pinned; serve without caching.
            return false;
        }
        self.insert_raw(id, bytes, fetch_cost);
        false
    }

    fn insert_raw(&mut self, id: ExpertId, bytes: u64, fetch_cost: f64) {
        let key = self.inflation + fetch_cost / bytes.max(1) as f64;
        self.entries.insert(
            id,
            Entry {
                bytes,
                fetch_cost,
                hits: 1,
                last_used: self.clock,
                key,
            },
        );
        self.used += bytes;
    }

    /// Evict until `bytes` will fit. Returns false if that is impossible.
    fn make_room(&mut self, bytes: u64) -> bool {
        while self.used + bytes > self.capacity {
            match self.pick_victim() {
                Some(v) => {
                    if let Some(e) = self.entries.remove(&v) {
                        self.used -= e.bytes;
                        self.stats.evictions += 1;
                        if self.policy == Policy::Gdsf {
                            // Carry the victim's key forward, so future entries
                            // are judged against what was given up. Without this
                            // an early-hot entry would keep its key for ever.
                            self.inflation = self.inflation.max(e.key);
                        }
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Choose the entry to evict.
    ///
    /// Ties are broken by expert id, deliberately. `entries` is a `HashMap`, and
    /// Rust randomises its hash seed per process, so scoring ties alone would
    /// make eviction — and therefore every reported hit rate — differ between
    /// runs. LFU is especially tie-heavy, since most entries sit at one hit.
    fn pick_victim(&self) -> Option<ExpertId> {
        let mut best: Option<(f64, ExpertId)> = None;
        for (id, e) in &self.entries {
            if self.pinned.contains_key(id) {
                continue;
            }
            let score = match self.policy {
                Policy::Lru => e.last_used as f64,
                Policy::Lfu => e.hits as f64,
                Policy::Gdsf => e.key,
            };
            let better = match best {
                None => true,
                Some((bs, bid)) => score < bs || (score == bs && *id < bid),
            };
            if better {
                best = Some((score, *id));
            }
        }
        best.map(|(_, id)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic Zipf-ish access stream: low ids are much hotter.
    fn skewed_stream(n: usize, n_experts: u32, seed: u64) -> Vec<ExpertId> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x >> 11) as f64 / (1u64 << 53) as f64;
                // u^3 concentrates mass near zero.
                let e = (u * u * u * f64::from(n_experts)) as u32;
                (0u32, e.min(n_experts - 1))
            })
            .collect()
    }

    #[test]
    fn a_hit_requires_a_prior_miss() {
        let mut c = ExpertCache::new(1000, Policy::Lru);
        assert!(!c.access((0, 1), 100, 1.0));
        assert!(c.access((0, 1), 100, 1.0));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn capacity_is_never_exceeded() {
        let mut c = ExpertCache::new(500, Policy::Gdsf);
        for e in 0..50u32 {
            c.access((0, e), 100, 1.0);
            assert!(
                c.used_bytes() <= c.capacity(),
                "used {} exceeded capacity {}",
                c.used_bytes(),
                c.capacity()
            );
        }
    }

    #[test]
    fn an_oversized_expert_is_served_but_not_retained() {
        let mut c = ExpertCache::new(100, Policy::Lru);
        assert!(!c.access((0, 1), 500, 1.0));
        assert!(!c.contains((0, 1)));
        assert_eq!(c.used_bytes(), 0);
        // The fetch is still accounted for.
        assert_eq!(c.stats().bytes_fetched, 500);
    }

    #[test]
    fn lru_evicts_the_stalest_entry() {
        let mut c = ExpertCache::new(300, Policy::Lru);
        c.access((0, 1), 100, 1.0);
        c.access((0, 2), 100, 1.0);
        c.access((0, 3), 100, 1.0);
        c.access((0, 1), 100, 1.0); // refresh 1
        c.access((0, 4), 100, 1.0); // must evict 2, the stalest
        assert!(!c.contains((0, 2)));
        assert!(c.contains((0, 1)));
    }

    #[test]
    fn lfu_keeps_the_hottest_entry() {
        let mut c = ExpertCache::new(300, Policy::Lfu);
        for _ in 0..5 {
            c.access((0, 1), 100, 1.0);
        }
        c.access((0, 2), 100, 1.0);
        c.access((0, 3), 100, 1.0);
        c.access((0, 4), 100, 1.0); // evicts 2 or 3, never the hot 1
        assert!(c.contains((0, 1)));
    }

    #[test]
    fn pinned_entries_survive_pressure() {
        let mut c = ExpertCache::new(300, Policy::Gdsf);
        assert!(c.pin((0, 99), 100, 1.0));
        for e in 0..20u32 {
            c.access((0, e), 100, 1.0);
        }
        assert!(c.contains((0, 99)), "a pinned expert was evicted");
        assert!(c.is_pinned((0, 99)));
    }

    #[test]
    fn a_fully_pinned_cache_stops_caching_rather_than_evicting() {
        let mut c = ExpertCache::new(200, Policy::Lru);
        assert!(c.pin((0, 1), 100, 1.0));
        assert!(c.pin((0, 2), 100, 1.0));
        // No room and nothing evictable: the access must still be served.
        assert!(!c.access((0, 3), 100, 1.0));
        assert!(!c.contains((0, 3)));
        assert!(c.contains((0, 1)) && c.contains((0, 2)));
    }

    #[test]
    fn pinning_beyond_capacity_is_refused_not_silently_dropped() {
        let mut c = ExpertCache::new(150, Policy::Lru);
        assert!(c.pin((0, 1), 100, 1.0));
        assert!(
            !c.pin((0, 2), 100, 1.0),
            "pin should fail when it cannot fit"
        );
        assert!(!c.is_pinned((0, 2)));
    }

    /// Best hit rate any policy could achieve with `resident` entries, computed
    /// offline: keep exactly the `resident` most-frequent experts.
    ///
    /// Comparing against this rather than an arbitrary threshold is what makes
    /// the test meaningful — it measures how close the online policy gets to the
    /// bound instead of asserting a number that happened to pass once.
    fn offline_optimal_hit_rate(stream: &[ExpertId], resident: usize) -> f64 {
        let mut counts: HashMap<ExpertId, u64> = HashMap::new();
        for id in stream {
            *counts.entry(*id).or_insert(0) += 1;
        }
        let mut v: Vec<u64> = counts.into_values().collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let head: u64 = v.iter().take(resident).sum();
        head as f64 / stream.len() as f64
    }

    #[test]
    fn skewed_access_approaches_the_offline_optimum() {
        // The property that makes a small resident set worth having: on skewed
        // routing a cache holding a quarter of the experts serves far more than a
        // quarter of accesses, and gets close to the offline bound.
        let stream = skewed_stream(20_000, 128, 5);
        let ceiling = offline_optimal_hit_rate(&stream, 32);
        let mut c = ExpertCache::new(32 * 1000, Policy::Gdsf);
        for id in &stream {
            c.access(*id, 1000, 1.0);
        }
        let got = c.stats().hit_rate();
        assert!(
            got > 0.25 * 1.8,
            "hit rate {got:.3} barely beat the uniform share of 0.25"
        );
        assert!(
            got >= ceiling * 0.90,
            "hit rate {got:.3} is under 90% of the offline optimum {ceiling:.3}"
        );
    }

    #[test]
    fn gdsf_gets_closest_to_the_offline_bound_at_uniform_cost() {
        // With identical size and cost there is nothing for cost-awareness to
        // exploit, so this isolates the value of the frequency term and the
        // inflation term together.
        //
        // Note that plain LFU does *not* beat LRU here. Unaged frequency counts
        // suffer the classical pollution problem: an entry that ran hot early
        // keeps its count for ever and cannot be displaced. GDSF's inflation
        // term is what fixes that, and it is the reason GDSF beats both parents
        // rather than sitting between them.
        let stream = skewed_stream(20_000, 128, 9);
        let bound = offline_optimal_hit_rate(&stream, 32);
        let run = |pol: Policy| -> CacheStats {
            let mut c = ExpertCache::new(32 * 1000, pol);
            for id in &stream {
                c.access(*id, 1000, 1.0);
            }
            c.stats()
        };
        let lru = run(Policy::Lru);
        let lfu = run(Policy::Lfu);
        let gdsf = run(Policy::Gdsf);

        assert!(
            gdsf.hit_rate() > lru.hit_rate() && gdsf.hit_rate() > lfu.hit_rate(),
            "gdsf {:.3} must beat lru {:.3} and lfu {:.3}",
            gdsf.hit_rate(),
            lru.hit_rate(),
            lfu.hit_rate()
        );
        assert!(
            gdsf.hit_rate() >= bound * 0.88,
            "gdsf {:.3} is far from the offline bound {bound:.3}",
            gdsf.hit_rate()
        );
        assert!(
            gdsf.fetch_seconds < lru.fetch_seconds,
            "gdsf spent {:.0}s vs lru {:.0}s",
            gdsf.fetch_seconds,
            lru.fetch_seconds
        );
    }

    #[test]
    fn gdsf_degenerates_to_lfu_when_cost_is_proportional_to_size() {
        // The limitation that matters in practice. With `cost = bytes / rate`,
        // GDSF's `cost / size` term is the same for every entry and contributes
        // nothing, so it must behave like LFU. Anyone choosing GDSF expecting a
        // free win on a single-tier store needs to know this.
        let stream = skewed_stream(20_000, 128, 9);
        let rate = 2.4e9f64;
        let bytes_of = |e: u32| if e % 2 == 0 { 1000u64 } else { 4000 };
        let run = |pol: Policy| -> CacheStats {
            let mut c = ExpertCache::new(40_000, pol);
            for id in &stream {
                let b = bytes_of(id.1);
                c.access(*id, b, b as f64 / rate);
            }
            c.stats()
        };
        let lfu = run(Policy::Lfu);
        let gdsf = run(Policy::Gdsf);
        assert!(
            (gdsf.hit_rate() - lfu.hit_rate()).abs() < 0.05,
            "with cost proportional to size GDSF should track LFU closely: \
             {:.3} vs {:.3}",
            gdsf.hit_rate(),
            lfu.hit_rate()
        );
    }

    #[test]
    fn gdsf_trades_hit_rate_for_time_when_costs_differ() {
        // The sharpest argument for cost-awareness, and a genuinely
        // counter-intuitive one: with mixed precisions GDSF ends up with a
        // *lower* hit rate than LFU and still spends less time fetching. It
        // keeps the expensive experts and lets cheap ones miss, because a miss on
        // a cheap expert costs little. Any system tuned on hit rate would call
        // this a regression; it is the opposite.
        let stream = skewed_stream(20_000, 128, 9);
        let cost_of = |e: u32| {
            if e % 2 == 0 {
                (1000u64, 1.0f64)
            } else {
                (4000, 8.0)
            }
        };
        let run = |pol: Policy| -> CacheStats {
            let mut c = ExpertCache::new(40_000, pol);
            for id in &stream {
                let (b, k) = cost_of(id.1);
                c.access(*id, b, k);
            }
            c.stats()
        };
        let lru = run(Policy::Lru);
        let lfu = run(Policy::Lfu);
        let gdsf = run(Policy::Gdsf);

        assert!(
            gdsf.fetch_seconds < lru.fetch_seconds && gdsf.fetch_seconds < lfu.fetch_seconds,
            "gdsf must minimise fetch time: {:.0}s vs lru {:.0}s, lfu {:.0}s",
            gdsf.fetch_seconds,
            lru.fetch_seconds,
            lfu.fetch_seconds
        );
        assert!(
            gdsf.hit_rate() < lfu.hit_rate(),
            "expected gdsf {:.3} to give up hit rate against lfu {:.3} in exchange for time; \
             if this no longer holds, the README's claim that hit rate is the wrong objective \
             needs restating",
            gdsf.hit_rate(),
            lfu.hit_rate()
        );
    }

    #[test]
    fn prefetch_does_not_inflate_the_hit_rate() {
        // Prefetching then reading must look like one hit, not two, and must not
        // register a miss it never suffered.
        let mut c = ExpertCache::new(1000, Policy::Gdsf);
        assert!(c.prefetch((0, 1), 100, 1.0));
        assert_eq!(c.stats().hits, 0);
        assert_eq!(c.stats().misses, 0);
        assert!(c.contains((0, 1)));
        assert!(c.access((0, 1), 100, 1.0));
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 0);
        // And no bytes were attributed here; the caller accounts for prefetches.
        assert_eq!(s.bytes_fetched, 0);
    }

    #[test]
    fn prefetching_something_resident_is_a_no_op() {
        let mut c = ExpertCache::new(1000, Policy::Lru);
        c.access((0, 1), 100, 1.0);
        assert!(!c.prefetch((0, 1), 100, 1.0));
        assert_eq!(c.used_bytes(), 100);
    }

    #[test]
    fn unpin_returns_an_entry_to_the_policy() {
        let mut c = ExpertCache::new(200, Policy::Lru);
        assert!(c.pin((0, 1), 100, 1.0));
        assert!(c.is_pinned((0, 1)));
        assert!(c.unpin((0, 1)));
        assert!(!c.is_pinned((0, 1)));
        // Now evictable.
        c.access((0, 2), 100, 1.0);
        c.access((0, 3), 100, 1.0);
        assert!(!c.contains((0, 1)), "an unpinned entry should be evictable");
    }

    #[test]
    fn results_are_reproducible_across_runs() {
        // Eviction must not depend on HashMap iteration order. Running the same
        // stream twice in one process exercises two different hash seeds only
        // across processes, so this checks the weaker but still useful property
        // that repeated identical runs agree, and the tie-break is total.
        let stream = skewed_stream(8_000, 64, 21);
        let run = || {
            let mut c = ExpertCache::new(16_000, Policy::Lfu);
            for id in &stream {
                c.access(*id, 1000, 1.0);
            }
            c.stats()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn stats_are_internally_consistent() {
        let stream = skewed_stream(5_000, 64, 3);
        let mut c = ExpertCache::new(20_000, Policy::Gdsf);
        for id in &stream {
            c.access(*id, 1000, 0.5);
        }
        let s = c.stats();
        assert_eq!(s.hits + s.misses, stream.len() as u64);
        assert_eq!(s.bytes_fetched, s.misses * 1000);
        assert!((s.fetch_seconds - s.misses as f64 * 0.5).abs() < 1e-9);
    }

    #[test]
    fn policy_names_round_trip() {
        for p in [Policy::Lru, Policy::Lfu, Policy::Gdsf] {
            assert_eq!(Policy::parse(p.name()), Some(p));
        }
        assert_eq!(Policy::parse("nope"), None);
    }
}
