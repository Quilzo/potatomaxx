// SPDX-License-Identifier: GPL-2.0-or-later
//! Commands for the streaming path: predictor evaluation, store construction,
//! and replay benchmarking.

use crate::moe;
use pmx_cache::Policy;
use pmx_gguf::Gguf;
use pmx_kernels::{ggml_dequant, PmxType, GROUP};
use pmx_plan::{ErrorBudget, PlanConfig, Sensitivity, Tier, TierCost};
use pmx_predict::{evaluate, Predictor};
use pmx_probe::Surface;
use pmx_runtime::{replay, RuntimeConfig};
use pmx_store::{Kind, Store, StoreWriter};
use pmx_trace::{CoActivation, Trace};

/// Format a byte count.
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
        format!("{v:.2} {}", U[i])
    }
}

/// Compare lookahead predictors on a trace.
pub fn predict(trace_path: &str, fit_frac: f64, budgets: &[usize]) -> Result<(), String> {
    let t = Trace::read(trace_path).map_err(|e| format!("reading {trace_path}: {e}"))?;
    let fit = ((t.n_tokens() as f64 * fit_frac) as usize).clamp(1, t.n_tokens().saturating_sub(1));
    let k = t.top_k as usize;

    println!("{trace_path}");
    println!(
        "  {} tokens, {} layers, {} experts, top-{k}",
        t.n_tokens(),
        t.n_layers,
        t.n_experts
    );
    println!(
        "  fitted on the first {fit} tokens, scored on the remaining {}",
        t.n_tokens() - fit
    );
    println!(
        "  chance recall at budget {k} is {:.3}\n",
        k as f64 / f64::from(t.n_experts)
    );
    println!(
        "{:>15} {:>7} {:>9} {:>10} {:>7}",
        "predictor", "budget", "recall", "precision", "fetch"
    );
    for &b in budgets {
        for p in Predictor::ALL {
            match evaluate(&t, p, fit, b) {
                Some(a) => println!(
                    "{:>15} {:>7} {:>9.3} {:>10.3} {:>6.2}x",
                    p.name(),
                    b,
                    a.recall,
                    a.precision,
                    a.fetch_amplification()
                ),
                None => println!("{:>15} {:>7}  (not scorable)", p.name(), b),
            }
        }
        println!();
    }
    println!(
        "recall is the share of the router's real choices a prefetcher would already\n\
         have had in flight. `fetch` is experts read per expert usefully obtained, so\n\
         it is the bandwidth premium paid for that recall."
    );
    println!(
        "\nthese predictors are training-free. Trained pre-attention routers reach\n\
         93-98% on real checkpoints; PILOT-style single-layer lookahead reports ~72%."
    );
    Ok(())
}

/// Build a native store from a GGUF checkpoint, requantising per expert.
#[allow(clippy::too_many_arguments)]
pub fn build_store(
    model: &str,
    trace_path: &str,
    out: &str,
    ram_mib: u64,
    group_align: u64,
    group_experts: u32,
    error_bits: f64,
    surface: &Surface,
) -> Result<(), String> {
    let g = Gguf::open(model).map_err(|e| format!("reading {model}: {e}"))?;
    let t = Trace::read(trace_path).map_err(|e| format!("reading {trace_path}: {e}"))?;
    let m = moe::detect(&g);
    if m.layers.is_empty() {
        return Err(format!("{model} has no MoE layers to store"));
    }

    // Tier costs come from the measured surface where available: a resident byte
    // is free, a stored byte costs what the device actually delivers.
    let storage_bps = surface
        .bandwidth_at(2 << 20, 8)
        .filter(|v| *v > 0.0)
        .unwrap_or(2.4e9);
    let tier_cost = TierCost {
        resident_bps: 30.0e9,
        storage_bps,
    };

    println!("building {out} from {model}");
    println!(
        "  storage bandwidth {:.2} GB/s ({}), resident {:.1} GB/s",
        storage_bps / 1e9,
        if surface.cells.is_empty() {
            "assumed"
        } else {
            "measured"
        },
        tier_cost.resident_bps / 1e9
    );

    let mut sw = StoreWriter::new(group_align, group_experts);
    let mut deq: Vec<f32> = Vec::new();
    let mut total_src = 0u64;
    let mut per_type: std::collections::BTreeMap<&'static str, u64> = Default::default();
    let mut skipped: Vec<String> = Vec::new();

    for (li, layer) in m.layers.iter().enumerate() {
        if li >= t.n_layers as usize {
            break;
        }
        let ca = CoActivation::from_trace(&t, li as u32);

        // Measure what each precision actually costs this layer's experts,
        // rather than assuming an analytic curve. The allocator's decisions are
        // only as good as this input, and an F16 checkpoint against the analytic
        // proxy produces no allocation at all.
        let mut per_expert_error: Vec<Vec<(f64, f64)>> =
            Vec::with_capacity(layer.n_experts as usize);
        for e in 0..layer.n_experts {
            let mut rows = vec![(layer.baseline_bits, 0.0f64)];
            // Sample one matrix: the three share a distribution closely enough
            // that measuring all of them triples the cost for no new signal.
            if let Some(name) = layer.expert_tensors.first() {
                if let Some(ti) = g.tensors.iter().find(|x| &x.name == name) {
                    if pmx_kernels::can_dequantize(ti.ggml_type.0) {
                        if let (Ok(raw), Ok(n)) = (g.read_tensor_slice(ti, e), g.slice_elements(ti))
                        {
                            let n = n as usize;
                            if n % GROUP == 0
                                && ggml_dequant::dequantize(ti.ggml_type.0, &raw, n, &mut deq)
                                    .is_ok()
                            {
                                for ty in PmxType::ALL {
                                    if let Ok(Some(rmse)) =
                                        pmx_kernels::pmxq::roundtrip_rmse(ty, &deq)
                                    {
                                        rows.push((ty.bits_per_weight(), rmse));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            per_expert_error.push(rows);
        }
        let sens = Sensitivity::measured(per_expert_error);
        // One allocation decision per expert, from this layer's observed
        // frequencies and the real tier costs.
        let cfg = PlanConfig {
            weights_per_expert: layer.weights_per_expert,
            resident_budget_bytes: (ram_mib << 20) / m.layers.len().max(1) as u64,
            baseline_bits: layer.baseline_bits,
            tier_cost,
            error_budget: ErrorBudget::UniformAt(error_bits),
            ..PlanConfig::default()
        };
        let alloc = pmx_plan::allocate(&ca, &sens, &cfg);

        for e in 0..layer.n_experts {
            let a = alloc.experts.get(e as usize);
            // Resident experts keep the most precision available; stored ones
            // take the allocator's choice. Cold and expensive is where bits are
            // worth giving up.
            let ty = match a {
                Some(x) if x.tier == Tier::Resident => PmxType::Q8,
                Some(x) => nearest_pmx_type(x.bits),
                None => PmxType::Q4,
            };
            for name in &layer.expert_tensors {
                let kind = match Kind::from_tensor_name(name) {
                    Some(k) => k,
                    None => continue,
                };
                let ti = match g.tensors.iter().find(|x| &x.name == name) {
                    Some(x) => x,
                    None => continue,
                };
                if !pmx_kernels::can_dequantize(ti.ggml_type.0) {
                    let msg = format!(
                        "{name}: {} cannot be decoded by this build",
                        ti.ggml_type.name()
                    );
                    if !skipped.contains(&msg) {
                        skipped.push(msg);
                    }
                    continue;
                }
                let raw = g
                    .read_tensor_slice(ti, e)
                    .map_err(|err| format!("{name} slice {e}: {err}"))?;
                let n = g
                    .slice_elements(ti)
                    .map_err(|err| format!("{name}: {err}"))? as usize;
                if n % GROUP != 0 {
                    let msg = format!(
                        "{name}: slice of {n} weights is not a multiple of the {GROUP}-weight group"
                    );
                    if !skipped.contains(&msg) {
                        skipped.push(msg);
                    }
                    continue;
                }
                ggml_dequant::dequantize(ti.ggml_type.0, &raw, n, &mut deq)
                    .map_err(|err| format!("{name} slice {e}: {err}"))?;
                sw.add(li as u32, e as u32, kind, ty, &deq)
                    .map_err(|err| format!("{name} slice {e}: {err}"))?;
                total_src += raw.len() as u64;
                *per_type.entry(ty.label()).or_insert(0) += 1;
            }
        }
        println!(
            "  block {:>3}: {} experts, {:.1}% resident, movement {:.2}x faster, \
             expected error {:.5} ({} sensitivity)",
            layer.block,
            layer.n_experts,
            pmx_plan::hit_rate(&alloc) * 100.0,
            alloc.speedup(),
            alloc.planned_loss,
            if sens.is_measured() {
                "measured"
            } else {
                "proxy"
            }
        );
    }

    if sw.is_empty() {
        let mut msg = String::from("nothing could be stored");
        for s in skipped.iter().take(4) {
            msg.push_str(&format!("\n  {s}"));
        }
        return Err(msg);
    }
    let stats = sw.finish(out).map_err(|e| format!("writing {out}: {e}"))?;
    println!(
        "\n  wrote {} slices: {} of weights plus {} alignment padding, {} groups",
        stats.records,
        human(stats.payload_bytes),
        human(stats.padding_bytes),
        stats.groups
    );
    println!("  source expert bytes read: {}", human(total_src));
    if total_src > 0 {
        println!(
            "  weights are {:.2}x the source for the same experts ({:.2}x including padding)",
            stats.payload_bytes as f64 / total_src as f64,
            stats.data_bytes as f64 / total_src as f64
        );
    }
    print!("  precision mix:");
    for (k, v) in &per_type {
        print!(" {k}={v}");
    }
    println!();
    if !skipped.is_empty() {
        println!("\n  skipped {} tensor(s):", skipped.len());
        for s in skipped.iter().take(6) {
            println!("    {s}");
        }
    }
    println!(
        "\nNote: routers are deliberately not requantised and stay in the GGUF.\n\
         Quantisation error in a router perturbs expert selection itself, which would\n\
         invalidate the trace this plan was derived from."
    );
    println!("\nnext:  potatomaxx bench --store {out} --trace {trace_path}");
    Ok(())
}

fn nearest_pmx_type(bits: f64) -> PmxType {
    let mut best = PmxType::Q4;
    let mut bd = f64::INFINITY;
    for ty in PmxType::ALL {
        let d = (ty.bits_per_weight() - bits).abs();
        if d < bd {
            bd = d;
            best = ty;
        }
    }
    best
}

/// Replay a trace against a store and report the fetch path.
#[allow(clippy::too_many_arguments)]
pub fn bench(
    store_path: &str,
    trace_path: &str,
    cache_mib: u64,
    policy: Policy,
    predictor: Option<Predictor>,
    budget: usize,
    queue_depth: usize,
    fit_frac: f64,
    surface: Surface,
    compare: bool,
) -> Result<(), String> {
    let s = Store::open(store_path).map_err(|e| format!("reading {store_path}: {e}"))?;
    let t = Trace::read(trace_path).map_err(|e| format!("reading {trace_path}: {e}"))?;
    let fit = ((t.n_tokens() as f64 * fit_frac) as usize).clamp(1, t.n_tokens().saturating_sub(1));

    println!("{store_path}");
    println!(
        "  {} slices, {} on disk, group alignment {}",
        s.records().len(),
        human(s.file_len()),
        human(s.group_align())
    );
    println!(
        "  replaying {} tokens ({} used to fit the predictor)\n",
        t.n_tokens() - fit,
        fit
    );

    let mk = |pred: Option<Predictor>, pol: Policy| RuntimeConfig {
        cache_bytes: cache_mib << 20,
        policy: pol,
        predictor: pred,
        prefetch_budget: budget,
        queue_depth,
        fit_tokens: fit,
        surface: surface.clone(),
    };

    let calibrated = !surface.cells.is_empty();
    if compare {
        let k = t.top_k as usize;
        println!("--- prefetch budget sweep (gdsf, sticky+markov) ---");
        println!(
            "{:>7} {:>10} {:>12} {:>11} {:>9}",
            "budget", "hit rate", "bytes/token", "useful", "tok/s"
        );
        let mut best_budget = (0usize, 0.0f64);
        for b in [0usize, k, k * 2, k * 3, k * 4] {
            let cfg = RuntimeConfig {
                prefetch_budget: b.max(1),
                predictor: if b == 0 {
                    None
                } else {
                    Some(Predictor::StickyMarkov)
                },
                ..mk(Some(Predictor::StickyMarkov), Policy::Gdsf)
            };
            let r = replay(&s, &t, &cfg);
            let tps = r.fetch_limited_tokens_per_sec();
            println!(
                "{:>7} {:>10.3} {:>12.0} {:>10.1}% {:>9.2}",
                if b == 0 {
                    "none".to_string()
                } else {
                    b.to_string()
                },
                r.cache.hit_rate(),
                r.bytes_per_token,
                r.prefetch_precision() * 100.0,
                tps
            );
            if tps > best_budget.1 {
                best_budget = (b, tps);
            }
        }
        println!(
            "\nbest budget here is {} at {:.2} tok/s. Prefetching is not free: every\n\
             prediction is a real read, so a budget far above top-{k} spends more bandwidth\n\
             than the queue depth wins back.\n",
            if best_budget.0 == 0 {
                "none".to_string()
            } else {
                best_budget.0.to_string()
            },
            best_budget.1
        );

        println!("--- policy and predictor matrix (budget {budget}) ---");
        println!(
            "{:>16} {:>7} {:>10} {:>12} {:>11} {:>9}",
            "configuration", "policy", "hit rate", "bytes/token", "req/token", "tok/s"
        );
        let mut rows: Vec<(String, f64)> = Vec::new();
        for pol in [Policy::Lru, Policy::Gdsf] {
            for pred in [None, Some(Predictor::Sticky), Some(Predictor::StickyMarkov)] {
                let r = replay(&s, &t, &mk(pred, pol));
                let label = pred.map(|p| p.name()).unwrap_or("on-demand");
                println!(
                    "{:>16} {:>7} {:>10.3} {:>12.0} {:>11.2} {:>9.2}",
                    label,
                    pol.name(),
                    r.cache.hit_rate(),
                    r.bytes_per_token,
                    r.requests_per_token,
                    r.fetch_limited_tokens_per_sec()
                );
                rows.push((
                    format!("{label}/{}", pol.name()),
                    r.fetch_limited_tokens_per_sec(),
                ));
            }
        }
        if let (Some(worst), Some(best)) = (
            rows.iter().min_by(|a, b| a.1.total_cmp(&b.1)),
            rows.iter().max_by(|a, b| a.1.total_cmp(&b.1)),
        ) {
            println!(
                "\nbest {} at {:.2} tok/s, worst {} at {:.2} — a spread of {:.1}x",
                best.0,
                best.1,
                worst.0,
                worst.1,
                best.1 / worst.1.max(1e-9)
            );
        }
    } else {
        let r = replay(&s, &t, &mk(predictor, policy));
        println!(
            "  cache            {} at {}",
            policy.name(),
            human(cache_mib << 20)
        );
        println!(
            "  predictor        {}",
            predictor.map(|p| p.name()).unwrap_or("none (on-demand)")
        );
        println!("  hit rate         {:.3}", r.cache.hit_rate());
        println!("  bytes/token      {}", human(r.bytes_per_token as u64));
        println!("  requests/token   {:.2}", r.requests_per_token);
        if r.useful_prefetches + r.wasted_prefetches > 0 {
            println!(
                "  prefetch useful  {:.1}% ({} used, {} wasted)",
                r.prefetch_precision() * 100.0,
                r.useful_prefetches,
                r.wasted_prefetches
            );
        }
        println!("  effective read   {:.2} GB/s", r.effective_bps / 1e9);
        println!(
            "  fetch-limited    {:.2} tok/s",
            r.fetch_limited_tokens_per_sec()
        );
    }
    if !calibrated {
        println!(
            "\nno --probe surface given, so times are request counts rather than seconds.\n\
             run `potatomaxx probe` first for calibrated throughput."
        );
    }
    println!(
        "\nThis is expert-fetch throughput, not generated tokens per second: the replay\n\
         moves weights but does not run attention or sampling. On a memory-bound machine\n\
         it is the binding constraint, and therefore a ceiling on decode rate."
    );
    Ok(())
}
