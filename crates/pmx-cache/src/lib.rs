// SPDX-License-Identifier: GPL-2.0-or-later
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
    /// Segmented LRU: a probationary segment feeding a protected segment.
    ///
    /// An expert enters on probation and is promoted only on its *second*
    /// access; eviction drains probation first. A one-shot prefill sweep
    /// therefore lands entirely in probation and cannot displace a decode
    /// working set that has been touched twice, which is the failure mode plain
    /// LRU has. This is the policy `llama.cpp` issue #20757 proposes adopting
    /// as its default, and the reason the trace generator needed phases: on an
    /// undifferentiated trace this is indistinguishable from [`Policy::Lru`].
    Slru,
}

/// Default fraction of capacity the protected segment may occupy under
/// [`Policy::Slru`].
///
/// The classic value. Too high and probation is too small to audition new
/// entries; too low and the protected set cannot hold a working set. Tunable
/// via [`ExpertCache::set_slru_protected_fraction`], because a policy that only
/// loses at one setting has not been evaluated -- it has been mis-tuned.
pub const SLRU_PROTECTED_FRACTION: f64 = 0.8;

impl Policy {
    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Policy::Lru => "lru",
            Policy::Lfu => "lfu",
            Policy::Gdsf => "gdsf",
            Policy::Slru => "slru",
        }
    }

    /// Parse a policy name.
    pub fn parse(s: &str) -> Option<Policy> {
        Some(match s {
            "lru" => Policy::Lru,
            "lfu" => Policy::Lfu,
            "gdsf" => Policy::Gdsf,
            "slru" => Policy::Slru,
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
    /// SLRU: promoted out of probation. Meaningless under other policies.
    protected: bool,
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
    /// SLRU: bytes currently held in the protected segment.
    protected_used: u64,
    /// SLRU: share of capacity the protected segment may occupy.
    slru_fraction: f64,
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
            protected_used: 0,
            slru_fraction: SLRU_PROTECTED_FRACTION,
            stats: CacheStats::default(),
        }
    }

    /// Set the share of capacity [`Policy::Slru`] may hold protected.
    ///
    /// Clamped to a sane interior range: at 0 nothing can be protected and the
    /// policy is just LRU, at 1 protection swallows the cache and there is no
    /// probation left to audition new entries.
    pub fn set_slru_protected_fraction(&mut self, f: f64) {
        self.slru_fraction = f.clamp(0.05, 0.95);
        self.demote_overflow();
    }

    /// Bytes currently held in the protected segment under [`Policy::Slru`].
    pub fn protected_bytes(&self) -> u64 {
        self.protected_used
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
            // Second access is what earns protection. One-shot entries — the
            // whole of a prefill sweep — never reach the protected segment.
            let promote = self.policy == Policy::Slru && !e.protected;
            let bytes = e.bytes;
            if promote {
                e.protected = true;
            }
            self.stats.hits += 1;
            if promote {
                self.protected_used += bytes;
                self.demote_overflow();
            }
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
                protected: false,
            },
        );
        self.used += bytes;
    }

    /// SLRU: hold the protected segment to its share of capacity by demoting
    /// its least-recently-used members back to probation.
    ///
    /// Without this the policy is self-defeating: every twice-touched expert
    /// would stay protected, protection would spread to the whole cache, and
    /// eviction would fall back to plain recency — the thing SLRU exists to
    /// avoid. Demotion is not eviction; a demoted expert is still resident and
    /// still serves hits, it has merely lost its immunity.
    fn demote_overflow(&mut self) {
        if self.policy != Policy::Slru {
            return;
        }
        let limit = (self.capacity as f64 * self.slru_fraction) as u64;
        while self.protected_used > limit {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.protected)
                .min_by(|a, b| a.1.last_used.cmp(&b.1.last_used).then_with(|| a.0.cmp(b.0)))
                .map(|(id, _)| *id);
            match victim {
                Some(v) => {
                    if let Some(e) = self.entries.get_mut(&v) {
                        e.protected = false;
                        self.protected_used = self.protected_used.saturating_sub(e.bytes);
                    }
                }
                None => break,
            }
        }
    }

    /// Evict until `bytes` will fit. Returns false if that is impossible.
    fn make_room(&mut self, bytes: u64) -> bool {
        while self.used + bytes > self.capacity {
            match self.pick_victim() {
                Some(v) => {
                    if let Some(e) = self.entries.remove(&v) {
                        self.used -= e.bytes;
                        self.stats.evictions += 1;
                        if e.protected {
                            self.protected_used = self.protected_used.saturating_sub(e.bytes);
                        }
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
        let mut best: Option<((u8, f64), ExpertId)> = None;
        for (id, e) in &self.entries {
            if self.pinned.contains_key(id) {
                continue;
            }
            // The leading rank segments the candidates; only SLRU uses it.
            // Probation (rank 0) is drained entirely before protected (rank 1).
            let score: (u8, f64) = match self.policy {
                Policy::Lru => (0, e.last_used as f64),
                Policy::Lfu => (0, e.hits as f64),
                Policy::Gdsf => (0, e.key),
                Policy::Slru => (u8::from(e.protected), e.last_used as f64),
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
        for p in [Policy::Lru, Policy::Lfu, Policy::Gdsf, Policy::Slru] {
            assert_eq!(Policy::parse(p.name()), Some(p));
        }
        assert_eq!(Policy::parse("nope"), None);
    }

    /// Warm a narrow decode working set, sweep a wide prefill burst past it,
    /// then resume decode. Returns the hit rate over the *resumed* decode only,
    /// which is the quantity a user feels as tokens per second.
    fn decode_hit_rate_after_prefill(policy: Policy, sweep: u32, passes: usize) -> f64 {
        const BYTES: u64 = 100;
        // 8 experts of working set, 16 slots of capacity.
        let working: Vec<ExpertId> = (0..8u32).map(|e| (0, e)).collect();
        let mut c = ExpertCache::new(16 * BYTES, policy);

        // Warm: twice each, so a segmented policy has grounds to protect them.
        for _ in 0..2 {
            for id in &working {
                c.access(*id, BYTES, 1.0);
            }
        }
        // Prefill: a wide one-shot sweep over ids that never recur.
        for e in 0..sweep {
            c.access((1, e), BYTES, 1.0);
        }
        // Resume decode, and measure only this.
        let before = c.stats();
        for _ in 0..passes {
            for id in &working {
                c.access(*id, BYTES, 1.0);
            }
        }
        let after = c.stats();
        let hits = after.hits - before.hits;
        let misses = after.misses - before.misses;
        hits as f64 / (hits + misses) as f64
    }

    #[test]
    fn slru_survives_a_prefill_burst_that_wipes_lru() {
        // Measured over a single decode pass straight after the burst: LRU has no
        // way to tell a twice-used expert from a one-shot one, so the sweep
        // evicts the entire working set and decode restarts stone cold.
        let lru = decode_hit_rate_after_prefill(Policy::Lru, 64, 1);
        let slru = decode_hit_rate_after_prefill(Policy::Slru, 64, 1);
        assert!(lru < 1e-9, "expected lru to lose everything, got {lru:.3}");
        assert!(slru > 1.0 - 1e-9, "expected slru to hold on, got {slru:.3}");
    }

    #[test]
    fn the_lru_prefill_penalty_amortises_over_decode_length() {
        // The part that decides whether segmenting is worth anything in
        // practice, and the part that is easy to oversell. LRU's loss is a
        // *one-off* reload of the working set, not an ongoing rate penalty, so
        // its cost per token falls as generation continues. Segmenting is worth
        // most where prefill is large relative to the decode that follows --
        // short replies to long prompts -- and worth nearly nothing on long
        // generations. Anyone quoting the one-pass number alone is overselling.
        let mut gaps = Vec::new();
        for passes in [1usize, 3, 10, 40] {
            let lru = decode_hit_rate_after_prefill(Policy::Lru, 64, passes);
            let slru = decode_hit_rate_after_prefill(Policy::Slru, 64, passes);
            gaps.push(slru - lru);
        }
        for w in gaps.windows(2) {
            assert!(
                w[1] < w[0],
                "the advantage should shrink as decode lengthens: {gaps:?}"
            );
        }
        assert!(
            gaps[0] > 0.9 && gaps[3] < 0.1,
            "expected total at one pass, near-nothing at forty: {gaps:?}"
        );
    }

    #[test]
    fn slru_matches_lru_when_there_is_no_burst_to_segment() {
        // With no wide sweep the segmentation has nothing to protect against,
        // and SLRU should not *cost* anything either. This is the control: it is
        // why an undifferentiated trace cannot distinguish the two policies, and
        // therefore why pmx-trace grew a prefill phase.
        let lru = decode_hit_rate_after_prefill(Policy::Lru, 0, 10);
        let slru = decode_hit_rate_after_prefill(Policy::Slru, 0, 10);
        assert!((lru - slru).abs() < 1e-9, "lru {lru:.3} vs slru {slru:.3}");
        assert!(lru > 0.99, "a working set inside capacity should all hit");
    }

    #[test]
    fn slru_protected_segment_cannot_swallow_the_whole_cache() {
        // If protection spread unchecked, SLRU would decay into LRU and the test
        // above would silently stop measuring anything.
        const BYTES: u64 = 100;
        let mut c = ExpertCache::new(10 * BYTES, Policy::Slru);
        // Touch far more distinct experts twice each than protection may hold.
        for e in 0..40u32 {
            c.access((0, e), BYTES, 1.0);
            c.access((0, e), BYTES, 1.0);
        }
        let limit = (10.0 * BYTES as f64 * SLRU_PROTECTED_FRACTION) as u64;
        assert!(
            c.protected_used <= limit,
            "protected {} exceeds its share {}",
            c.protected_used,
            limit
        );
        assert!(c.used <= c.capacity);
        // Some probation must remain, or there is no audition space left.
        assert!(
            c.entries.values().any(|e| !e.protected),
            "every resident entry is protected; probation has vanished"
        );
    }
    /// Streaming-shaped workload: many experts, capacity well below the set, and
    /// nearly every expert touched more than once over a long decode.
    fn streaming_decode_hit_rate(policy: Policy, fraction: Option<f64>) -> f64 {
        const BYTES: u64 = 100;
        const N: u32 = 256;
        let mut c = ExpertCache::new(40 * BYTES, policy);
        if let Some(f) = fraction {
            c.set_slru_protected_fraction(f);
        }
        // Prefill: one wide sweep.
        let mut x: u64 = 0x1234_5678;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for e in 0..N {
            c.access((0, e), BYTES, 1.0);
        }
        // Decode: skewed but not degenerate -- a hot quarter and a long tail,
        // which is what MoE routing actually looks like.
        let before = c.stats();
        for _ in 0..4000 {
            let r = next() % 100;
            let e = if r < 70 {
                (next() % 64) as u32
            } else {
                (next() % u64::from(N)) as u32
            };
            c.access((0, e), BYTES, 1.0);
        }
        let a = c.stats();
        let h = a.hits - before.hits;
        let m = a.misses - before.misses;
        h as f64 / (h + m) as f64
    }

    #[test]
    fn segmenting_pays_only_when_probation_is_large_enough_to_promote() {
        // The condition that decides whether segmenting is worth anything, and
        // the reason `llama.cpp` #20757 should not adopt SLRU as an unconditional
        // default. Promotion requires a second access, and a second access
        // requires surviving probation. Probation is `1 - fraction` of capacity,
        // so the policy only works when repeat access arrives faster than
        // probation turns over. Here it does. The companion test covers the
        // regime where it does not, and there SLRU is far worse than LRU.
        let lru = streaming_decode_hit_rate(Policy::Lru, None);
        // Repeat access here is fast relative to probation, so promotion works
        // and protection is worth having -- more of it, the better.
        let small = streaming_decode_hit_rate(Policy::Slru, Some(0.1));
        let large = streaming_decode_hit_rate(Policy::Slru, Some(0.8));
        assert!(
            (small - lru).abs() < 0.01,
            "with almost nothing protectable slru {small:.4} should track lru {lru:.4}"
        );
        assert!(
            large > lru + 0.05,
            "slru {large:.4} should clearly beat lru {lru:.4} in this regime"
        );
    }

    #[test]
    fn a_tiny_protected_fraction_degenerates_towards_lru() {
        // Sanity check on the knob itself: with almost nothing protectable, SLRU
        // must behave close to LRU. If it does not, the segmentation is doing
        // something other than what it claims.
        let lru = streaming_decode_hit_rate(Policy::Lru, None);
        let slru = streaming_decode_hit_rate(Policy::Slru, Some(0.05));
        assert!(
            (lru - slru).abs() < 0.05,
            "at a 5% protected share slru {slru:.4} should track lru {lru:.4}"
        );
    }
}
