// SPDX-License-Identifier: GPL-2.0-or-later
//! `potatomaxx` — a layout compiler for disk-resident mixture-of-experts models.
//!
//! Reads a GGUF MoE checkpoint and a routing trace, works out an expert order
//! that turns scattered slice reads into contiguous runs, and rewrites the file.
//! The output is a drop-in GGUF: same tensor names, same shapes, byte-identical
//! weights, different offsets.
//!
//! It also reports, honestly, when there is nothing to gain.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod advise_cmd;
mod compare_cmd;
mod engine_cmds;
mod kio_cmd;
mod moe;
/// Half-precision encoding, re-exported for the synthetic model writer.
mod synth_half {
    pub use pmx_kernels::half::f32_to_f16;
}
mod planfile;
mod synth;

use planfile::{LayerPlan, Plan};
use pmx_gguf::{Gguf, PermSpec, Placement};
use pmx_partition::{CostModel, Edges, OptimizeConfig};
use pmx_plan::{PlanConfig, Sensitivity, TierCost};
use pmx_probe::{ProbeConfig, Surface};
use pmx_trace::{CoActivation, Trace};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

/// Print a line to stdout, ignoring a closed pipe.
///
/// `println!` panics when stdout has gone away, which makes piping into `head`
/// look like a crash. A command-line tool should simply stop writing.
macro_rules! outln {
    () => {{
        let _ = writeln!(std::io::stdout().lock());
    }};
    ($($arg:tt)*) => {{
        let _ = writeln!(std::io::stdout().lock(), $($arg)*);
    }};
}

/// As [`outln`], without the newline.
macro_rules! out {
    ($($arg:tt)*) => {{
        let _ = write!(std::io::stdout().lock(), $($arg)*);
    }};
}

const USAGE: &str = "\
potatomaxx — run frontier MoE models on a potato

USAGE
  potatomaxx <command> [options]

COMMANDS
  advise       Say which optimisations are worth attempting on this model and device
  probe        Measure this device's read bandwidth surface (blob size x queue depth)
  kio          Compare kernel I/O paths: pread, thread pool, io_uring, RWF_DONTCACHE
  predict      Compare router-lookahead predictors on a trace
  build-store  Requantise a checkpoint per expert into a native .pmxstore
  bench        Replay a trace against a store, measuring the fetch path
  synth        Write a small synthetic MoE checkpoint and trace, for trying the pipeline
  inspect      Report the MoE structure of a GGUF checkpoint
  fit          Will this model load in this RAM? (+ --recommend-quant, --json)
  compare      Check two quantisations of the same model dequantise to the same weights
  analyse      Score the existing expert order against an optimised one, per layer
  plan         Write a repack plan
  pack         Apply a plan, emitting a new GGUF
  verify       Prove a repacked file holds exactly the original weights

OPTIONS
  probe    --dir <path>          directory to test (default .)
           --corpus-mib <n>      scratch corpus size (default 2048)
           --out <path>           where to write the surface (default pmx-probe.json)

  synth    --out <path>          model path (default synth.gguf)
           --trace <path>        trace path (default synth.pmxtrace)
           --experts <n>         experts per layer (default 32)
           --layers <n>          MoE layers (default 2)
           --tokens <n>          tokens to simulate (default 4000)
           --clusters <n>        planted co-activation clusters (default 4)
           --locality <0..1>     share of tokens drawn from one cluster (default 0.85)
                                 — this is *within-token* co-activation
           --persistence <0..1>  chance a token reuses the previous token's cluster
                                 (default 0.7) — *across-token* structure, which is
                                 what lookahead prediction needs
           --layer-coupling <0..1>
                                 chance a layer reuses the previous layer's cluster
                                 (default 0.45) — what makes layer n predictable
                                 from layer n-1, and so prefetchable
           --no-scatter          keep planted clusters on contiguous expert ids
                                 (unrealistic: makes the existing order optimal)

  advise   --model <path> [--trace <path>] [--probe <path>] [--cache-mib <n>]
           [--acceptance <0..1>] draft-token acceptance rate for the speculative
                                 decoding estimate (default 0.7). Measure it;
                                 the whole gain scales with it.
           Runs every check the inputs allow and ranks the results: hazards
           first, then what is worth doing, then what measurably is not.

  inspect  <model.gguf>

  compare  --a <reference.gguf> --b <candidate.gguf> [--tensor <substr>] [--limit <n>]
           Dequantises tensors common to both and reports agreement. Use a
           higher-precision file as --a: an F16 reference validates a quantised
           decoder against ground truth rather than against itself.

  analyse  --model <path> --trace <path> [--probe <path>] [--merge-gap <n>]
           [--queue-depth <n>]   reads the consuming runtime keeps in flight (default 8)
  plan     --model <path> --trace <path> [--probe <path>] [--merge-gap <n>]
           [--queue-depth <n>] [--ram-mib <n>] [--min-speedup <f>] [--out <path>]
           --min-speedup <f>     gain a layer must clear to be repacked (default 1.05)
  pack     --model <path> --plan <path> --out <path>
  verify   --model <path> --repacked <path> --plan <path>

  kio      [--dir <path>] [--corpus-mib <n>] [--blob-kib <n>] [--ms <n>]

  predict  --trace <path> [--fit <0..1>] [--budgets <a,b,c>]

  build-store --model <path> --trace <path> --out <path> [--probe <path>]
           [--ram-mib <n>]       RAM budget for resident experts (default 512)
           [--group-align <n>]   group alignment in bytes (default 2097152)
           [--group-experts <n>] experts per aligned group (default 8)
           [--error-bits <f>]    quality ceiling, expressed as the bit width whose
                                 uniform error must not be exceeded (default 4.5).
                                 Lower means more aggressive.

  bench    --store <path> --trace <path> [--probe <path>]
           [--cache-mib <n>] [--policy lru|lfu|gdsf] [--predictor <name>]
           [--budget <n>] [--queue-depth <n>] [--fit <0..1>] [--compare]

NOTE
  Layout is the secondary lever. On the development machine, coalescing reads is
  worth 1.1-1.6x, while raising I/O queue depth is worth 7-8x — and queue depth
  belongs to an inference runtime, not to a file. potatomaxx does the part a file
  can do, and tells you how much that is.
";

struct Args {
    cmd: String,
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Option<Args> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next()?;
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut pending: Option<String> = None;
        for a in it {
            if let Some(k) = pending.take() {
                flags.push((k, a));
                continue;
            }
            if let Some(k) = a.strip_prefix("--") {
                if let Some((k, v)) = k.split_once('=') {
                    flags.push((k.to_string(), v.to_string()));
                } else {
                    pending = Some(k.to_string());
                }
            } else {
                positional.push(a);
            }
        }
        if let Some(k) = pending {
            flags.push((k, "true".to_string()));
        }
        Some(Args {
            cmd,
            positional,
            flags,
        })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn req(&self, key: &str) -> Result<&str, String> {
        self.get(key)
            .ok_or_else(|| format!("missing required --{key}"))
    }

    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{key} expects a number, got {v:?}")),
        }
    }
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Some(a) => a,
        None => {
            out!("{USAGE}");
            return ExitCode::SUCCESS;
        }
    };
    let r = match args.cmd.as_str() {
        "probe" => cmd_probe(&args),
        "synth" => cmd_synth(&args),
        "inspect" => cmd_inspect(&args),
        "fit" => cmd_fit(&args),
        "analyse" | "analyze" => cmd_analyse(&args),
        "plan" => cmd_plan(&args),
        "pack" => cmd_pack(&args),
        "verify" => cmd_verify(&args),
        "advise" => cmd_advise(&args),
        "kio" => cmd_kio(&args),
        "compare" => cmd_compare(&args),
        "predict" => cmd_predict(&args),
        "build-store" => cmd_build_store(&args),
        "bench" => cmd_bench(&args),
        "help" | "-h" | "--help" => {
            out!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
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
        format!("{v:.2} {}", U[i])
    }
}

fn load_surface(args: &Args) -> Surface {
    match args.get("probe") {
        Some(p) => match std::fs::read_to_string(p)
            .ok()
            .and_then(|t| Surface::from_json(&t))
        {
            Some(s) => {
                if !s.cache_bypassed {
                    eprintln!(
                        "warning: {p} was measured without page-cache bypass; predicted times \
                         will be optimistic"
                    );
                }
                s
            }
            None => {
                eprintln!("warning: could not read a surface from {p}; costs will be uncalibrated");
                Surface::default()
            }
        },
        None => {
            eprintln!(
                "note: no --probe surface given; reporting request counts only.\n      \
                 run `potatomaxx probe` first for calibrated timings."
            );
            Surface::default()
        }
    }
}

fn parse_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| {
            x.parse::<usize>()
                .map_err(|_| format!("{x:?} is not a number"))
        })
        .collect()
}

fn cmd_compare(args: &Args) -> Result<(), String> {
    compare_cmd::run(
        args.req("a")?,
        args.req("b")?,
        args.get("tensor"),
        args.num("limit", 12)?,
    )
}

fn cmd_advise(args: &Args) -> Result<(), String> {
    let surface = load_surface(args);
    advise_cmd::run(
        args.req("model")?,
        args.get("trace"),
        &surface,
        args.num("cache-mib", 64)?,
        args.num("acceptance", 0.7)?,
    )
}

fn cmd_kio(args: &Args) -> Result<(), String> {
    kio_cmd::run(
        args.get("dir").unwrap_or("."),
        args.num("corpus-mib", 1024)?,
        args.num("blob-kib", 64)?,
        args.num("ms", 400)?,
    )
}

fn cmd_predict(args: &Args) -> Result<(), String> {
    let trace = args.req("trace")?;
    let fit: f64 = args.num("fit", 0.5)?;
    let budgets = match args.get("budgets") {
        Some(v) => parse_list(v)?,
        None => {
            let t = Trace::read(trace).map_err(|e| format!("reading {trace}: {e}"))?;
            let k = t.top_k as usize;
            vec![k, k * 2, k * 4]
        }
    };
    engine_cmds::predict(trace, fit, &budgets)
}

fn cmd_build_store(args: &Args) -> Result<(), String> {
    let model = args.req("model")?;
    let trace = args.req("trace")?;
    let out = args.req("out")?;
    let ram_mib: u64 = args.num("ram-mib", 512)?;
    let group_align: u64 = args.num("group-align", 2 << 20)?;
    let group_experts: u32 = args.num("group-experts", 8)?;
    let error_bits: f64 = args.num("error-bits", 4.5)?;
    let surface = load_surface(args);
    engine_cmds::build_store(
        model,
        trace,
        out,
        ram_mib,
        group_align,
        group_experts,
        error_bits,
        &surface,
    )
}

fn cmd_bench(args: &Args) -> Result<(), String> {
    let store = args.req("store")?;
    let trace = args.req("trace")?;
    let cache_mib: u64 = args.num("cache-mib", 64)?;
    let policy = match args.get("policy") {
        Some(p) => pmx_cache::Policy::parse(p)
            .ok_or_else(|| format!("unknown policy {p:?}; try lru, lfu or gdsf"))?,
        None => pmx_cache::Policy::Gdsf,
    };
    let predictor = match args.get("predictor") {
        Some("none") => None,
        Some(p) => Some(
            pmx_predict::Predictor::parse(p).ok_or_else(|| format!("unknown predictor {p:?}"))?,
        ),
        None => Some(pmx_predict::Predictor::StickyMarkov),
    };
    let budget: usize = args.num("budget", 16)?;
    let queue_depth: usize = args.num("queue-depth", 8)?;
    let fit: f64 = args.num("fit", 0.5)?;
    let surface = load_surface(args);
    engine_cmds::bench(
        store,
        trace,
        cache_mib,
        policy,
        predictor,
        budget,
        queue_depth,
        fit,
        surface,
        args.get("compare").is_some(),
    )
}

fn cmd_probe(args: &Args) -> Result<(), String> {
    let dir = PathBuf::from(args.get("dir").unwrap_or("."));
    let corpus_mib: u64 = args.num("corpus-mib", 2048)?;
    let out = args.get("out").unwrap_or("pmx-probe.json");
    let cfg = ProbeConfig {
        dir: dir.clone(),
        corpus_bytes: corpus_mib << 20,
        ..Default::default()
    };

    outln!(
        "probing {} with a {corpus_mib} MiB corpus (O_DIRECT, random offsets)",
        dir.display()
    );
    outln!("this writes and then deletes a scratch file; it does not touch anything else\n");
    let s = pmx_probe::measure(&cfg).map_err(|e| format!("probe failed: {e}"))?;

    let qds: Vec<usize> = {
        let mut v: Vec<usize> = s.cells.iter().map(|c| c.queue_depth).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let blobs: Vec<u64> = {
        let mut v: Vec<u64> = s.cells.iter().map(|c| c.blob_bytes).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    out!("{:>10}", "blob");
    for q in &qds {
        out!("{:>9}", format!("QD{q}"));
    }
    outln!();
    for b in &blobs {
        out!("{:>10}", human(*b));
        for q in &qds {
            let v = s
                .cells
                .iter()
                .find(|c| c.blob_bytes == *b && c.queue_depth == *q)
                .map(|c| c.bytes_per_sec / 1e9)
                .unwrap_or(f64::NAN);
            out!("{v:>9.2}");
        }
        outln!("  GB/s");
    }
    if !s.cache_bypassed {
        outln!(
            "\nWARNING: this platform has no page-cache bypass, so these figures include\n\
             cached reads and overstate the device. Treat the shape as indicative and the\n\
             absolute numbers as an upper bound."
        );
    }
    let peak = s.peak() / 1e9;
    let worst = s
        .cells
        .iter()
        .map(|c| c.bytes_per_sec)
        .fold(f64::INFINITY, f64::min)
        / 1e9;
    outln!(
        "\npeak {peak:.2} GB/s, worst {worst:.2} GB/s — a spread of {:.1}x",
        peak / worst.max(1e-9)
    );
    outln!("the worst cell is the access pattern of a demand-paged mmap");
    std::fs::write(out, s.to_json()).map_err(|e| format!("writing {out}: {e}"))?;
    outln!("\nwrote {out}");
    Ok(())
}

fn cmd_synth(args: &Args) -> Result<(), String> {
    let model = args.get("out").unwrap_or("synth.gguf");
    let tracep = args.get("trace").unwrap_or("synth.pmxtrace");
    let experts: u32 = args.num("experts", 32)?;
    let layers: u32 = args.num("layers", 2)?;
    let tokens: usize = args.num("tokens", 4000)?;
    let clusters: u32 = args.num("clusters", 4)?;
    let locality: f64 = args.num("locality", 0.85)?;
    let top_k: u32 = args.num("top-k", 4)?;

    let shape = synth::SynthShape {
        layers,
        experts,
        ..Default::default()
    };
    let n = synth::write_gguf(model, shape).map_err(|e| format!("writing {model}: {e}"))?;
    outln!(
        "wrote {model} ({}) — {layers} MoE layers x {experts} experts",
        human(n)
    );

    let persistence: f64 = args.num("persistence", 0.7)?;
    let layer_coupling: f64 = args.num("layer-coupling", 0.45)?;
    let prefill_tokens: usize = args.num("prefill", 0)?;
    let prefill_locality: f64 = args.num("prefill-locality", 0.05)?;
    let mut t = Trace::synthetic_cfg(&pmx_trace::SynthConfig {
        n_layers: layers,
        n_experts: experts,
        top_k,
        tokens,
        clusters,
        locality,
        persistence,
        layer_coupling,
        prefill_tokens,
        prefill_locality,
        seed: 0xC0FFEE,
    });
    // Planted clusters land on contiguous ids, which would make the checkpoint's
    // existing order already optimal. Real expert numbering has no such
    // locality, so scatter the labels unless asked not to.
    if args.get("no-scatter").is_none() {
        t.scatter_labels(0x5EED);
    }
    t.write(tracep)
        .map_err(|e| format!("writing {tracep}: {e}"))?;
    outln!(
        "wrote {tracep} — {tokens} tokens, top-{top_k}, {clusters} clusters, \
         locality {locality}, persistence {persistence}, layer-coupling {layer_coupling}"
    );
    let ca = CoActivation::from_trace(&t, 0);
    outln!(
        "layer 0 skew: top {} of {experts} experts take {:.1}% of selections (uniform would be {:.1}%)",
        experts / 4,
        ca.mass_in_top((experts / 4) as usize) * 100.0,
        25.0
    );
    outln!("\nnext:  potatomaxx analyse --model {model} --trace {tracep}");
    Ok(())
}

fn open_model(path: &str) -> Result<Gguf, String> {
    Gguf::open(path).map_err(|e| format!("reading {path}: {e}"))
}

fn cmd_inspect(args: &Args) -> Result<(), String> {
    let path = args
        .positional
        .first()
        .map(|s| s.as_str())
        .or_else(|| args.get("model"))
        .ok_or("usage: potatomaxx inspect <model.gguf>")?;
    let g = open_model(path)?;
    outln!("{path}");
    outln!("  version      {}", g.version);
    outln!("  architecture {}", g.architecture().unwrap_or("(unset)"));
    outln!("  alignment    {} B", g.alignment);
    outln!("  tensors      {}", g.tensors.len());
    outln!("  metadata     {} keys", g.kvs.len());
    outln!(
        "  data section {} at offset {}",
        human(g.data_len()),
        g.data_start
    );

    let m = moe::detect(&g);
    if m.layers.is_empty() {
        outln!("\nno permutable MoE layers found — potatomaxx has nothing to do here");
    } else {
        outln!("\n{} MoE layers", m.layers.len());
        let l = &m.layers[0];
        outln!(
            "  experts/layer      {}\n  slice (largest)    {}\n  bytes/expert       {}\n  weights/expert     {}\n  precision          {} ({:.2} bits/weight)",
            l.n_experts,
            human(l.slice_bytes),
            human(l.bytes_per_expert),
            l.weights_per_expert,
            l.baseline_type,
            l.baseline_bits
        );
        outln!("  expert-axis tensors per layer: {}", l.tensors.len());
        for t in &l.tensors {
            outln!("    {t}");
        }
        let total: u64 = m
            .layers
            .iter()
            .map(|l| l.bytes_per_expert * l.n_experts)
            .sum();
        outln!("  expert weight total: {}", human(total));
    }
    if !m.skipped.is_empty() {
        outln!("\nskipped {} expert-shaped tensors:", m.skipped.len());
        for (n, why) in m.skipped.iter().take(8) {
            outln!("  {n}: {why}");
        }
    }
    Ok(())
}

/// Total system RAM in bytes, from `/proc/meminfo` (Linux). None if unreadable.
fn total_ram_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// KV cache + compute-buffer + OS reserve. Weights dominate; the ctx term is a
/// modest linear allowance. Calibrated so the verdict matches llama.cpp's own fit
/// check (which aborted an ~11 GiB model on ~7.9 GiB RAM).
fn fit_reserve(ctx: u64) -> u64 {
    (400u64 << 20) + ctx * (128 << 10)
}

/// Usable share of RAM before the runtime refuses to load (leaves headroom for
/// the OS and allocator).
fn fit_ceiling(ram_bytes: u64) -> u64 {
    (ram_bytes as f64 * 0.95) as u64
}

/// Standard GGUF quant tiers and their approximate effective bits-per-weight
/// (calibrated to observed k-/i-quant file sizes), ascending.
const QUANT_TIERS: &[(&str, f64)] = &[
    ("IQ1_S", 1.70),
    ("IQ2_XXS", 2.06),
    ("Q2_K", 2.95),
    ("IQ3_XXS", 3.05),
    ("Q3_K_M", 3.55),
    ("Q4_K_M", 4.87),
    ("Q5_K_M", 5.65),
    ("Q6_K", 6.58),
    ("Q8_0", 8.55),
    ("F16", 16.0),
];

/// Approximate on-disk/in-RAM weight bytes for `params` at `bpw` bits/weight.
fn quant_weight_bytes(params: u64, bpw: f64) -> u64 {
    (params as f64 * bpw / 8.0) as u64
}

/// The largest quant tier whose weights fit `budget_weights`, if any.
fn recommend_quant(params: u64, budget_weights: u64) -> Option<(&'static str, u64)> {
    let mut best = None;
    for (name, bpw) in QUANT_TIERS {
        let sz = quant_weight_bytes(params, *bpw);
        if sz <= budget_weights {
            best = Some((*name, sz));
        }
    }
    best
}

/// `fit`: will this GGUF load in a given RAM budget, and if not, what to target.
///
/// The point that costs people hours: a model whose weights exceed RAM does not
/// merely run slowly — llama.cpp/Ollama pre-check the fit and ABORT the load. mmap
/// does not lift that on a RAM-bound host (every token of a dense model, and the
/// full resident set of a MoE, must be backable). So the honest question before a
/// multi-GB download is "does it fit", not "how fast".
fn cmd_fit(args: &Args) -> Result<(), String> {
    let path = args
        .positional
        .first()
        .map(|s| s.as_str())
        .or_else(|| args.get("model"))
        .ok_or("usage: potatomaxx fit <model.gguf> [--ram <GiB>] [--ctx <n>] [--recommend-quant] [--json]")?;
    let g = open_model(path)?;
    let data = g.data_len();
    let m = moe::detect(&g);

    let ram_bytes: u64 = if args.get("ram").is_some() {
        let gib: f64 = args.num("ram", 0.0)?;
        (gib * (1u64 << 30) as f64) as u64
    } else {
        total_ram_bytes().unwrap_or(0)
    };
    let ctx: u64 = args.num("ctx", 4096)?;

    let reserve = fit_reserve(ctx);
    let projected = data + reserve;

    if args.get("json").is_some() {
        let ceiling = fit_ceiling(ram_bytes);
        let fits = ram_bytes > 0 && projected <= ceiling;
        let max_weights = ceiling.saturating_sub(reserve);
        outln!(
            "{{\"weights_bytes\":{},\"reserve_bytes\":{},\"projected_bytes\":{},\"ram_bytes\":{},\"ctx\":{},\"fits\":{},\"max_weights_bytes\":{},\"is_moe\":{}}}",
            data, reserve, projected, ram_bytes, ctx, fits, max_weights, !m.layers.is_empty()
        );
        return Ok(());
    }

    outln!("{path}");
    outln!("  weights (data)   {}", human(data));
    outln!("  reserve (kv+ctx) {}  (ctx={ctx})", human(reserve));
    outln!("  projected use    {}", human(projected));
    if ram_bytes == 0 {
        outln!("  system RAM       (unknown — pass --ram <GiB>)");
        return Ok(());
    }
    outln!("  system RAM       {}", human(ram_bytes));

    if args.get("recommend-quant").is_some() || args.get("recommend").is_some() {
        let params: u64 = g.tensors.iter().map(|t| t.n_elements().unwrap_or(0)).sum();
        let budget = fit_ceiling(ram_bytes).saturating_sub(reserve);
        outln!("  parameters       ~{:.1}B", params as f64 / 1e9);
        outln!("  weight budget    {}\n", human(budget));
        outln!("  quant      bpw    ~weights   fits");
        for (name, bpw) in QUANT_TIERS {
            let sz = quant_weight_bytes(params, *bpw);
            outln!(
                "  {:<9} {:>4.2}  {:>9}   {}",
                name,
                bpw,
                human(sz),
                if sz <= budget { "yes" } else { "no" }
            );
        }
        match recommend_quant(params, budget) {
            Some((n, sz)) => outln!(
                "\nRecommended: {} (~{} weights) — largest quant that fits {} RAM.",
                n,
                human(sz),
                human(ram_bytes)
            ),
            None => outln!(
                "\nNo standard quant fits {} RAM — use a smaller model or raise RAM.",
                human(ram_bytes)
            ),
        }
        return Ok(());
    }

    let ceiling = fit_ceiling(ram_bytes);
    if projected <= ceiling {
        outln!(
            "\nFITS — projected {} within {} usable RAM.",
            human(projected),
            human(ceiling)
        );
    } else {
        let deficit = projected - ceiling;
        let max_weights = ceiling.saturating_sub(reserve);
        outln!(
            "\nDOES NOT FIT — projected {} exceeds usable RAM by {}.",
            human(projected),
            human(deficit)
        );
        outln!(
            "  target a quant whose weights are <= {} (or raise RAM).",
            human(max_weights)
        );
    }

    if !m.layers.is_empty() {
        outln!(
            "\nnote: MoE ({} experts/layer). Only ~active experts compute per token, but the",
            m.layers[0].n_experts
        );
        outln!("      runtime still requires the FULL model to fit RAM and aborts otherwise —");
        outln!("      mmap does not lift that on a RAM-bound host. `build-store` can shrink the");
        outln!("      resident set, but needs potatomaxx's own runtime to serve inference.");
    }
    Ok(())
}

/// One layer's detected structure paired with what the optimiser made of it.
type LayerAnalysis = (moe::MoeLayer, pmx_partition::OptimizeReport);

/// The result of analysing a checkpoint against a trace.
struct Analysis {
    trace: Trace,
    layers: Vec<LayerAnalysis>,
    /// Speedup a layer must clear to be judged worth repacking.
    min_speedup: f64,
}

/// Shared analysis core: for each layer, optimise and report.
fn analyse_layers(args: &Args) -> Result<Analysis, String> {
    let model = args.req("model")?;
    let tracep = args.req("trace")?;
    let merge_gap: u32 = args.num("merge-gap", 0)?;
    let queue_depth: usize = args.num("queue-depth", 8)?;
    let min_speedup: f64 = args.num("min-speedup", 1.05)?;
    let surface = load_surface(args);

    let g = open_model(model)?;
    let t = Trace::read(tracep).map_err(|e| format!("reading {tracep}: {e}"))?;
    let m = moe::detect(&g);
    if m.layers.is_empty() {
        return Err(format!("{model} has no permutable MoE layers"));
    }

    let mut out = Vec::new();
    for layer in m.layers {
        // A trace layer index need not equal the block index; map by position.
        let trace_layer = out.len() as u32;
        if trace_layer >= t.n_layers {
            break;
        }
        if u64::from(t.n_experts) != layer.n_experts {
            return Err(format!(
                "trace describes {} experts but block {} has {}",
                t.n_experts, layer.block, layer.n_experts
            ));
        }
        let edges = Edges::from_trace(&t, trace_layer);
        let ca = CoActivation::from_trace(&t, trace_layer);
        let cm = CostModel {
            slice_bytes: layer.slice_bytes,
            tensors_per_expert: layer.expert_tensors.len() as u32,
            merge_gap_slices: merge_gap,
            queue_depth,
            surface: surface.clone(),
        };
        let rep = pmx_partition::optimize(&edges, &ca, &cm, &OptimizeConfig::default());
        out.push((layer, rep));
    }
    Ok(Analysis {
        trace: t,
        layers: out,
        min_speedup,
    })
}

fn cmd_analyse(args: &Args) -> Result<(), String> {
    let calibrated = args.get("probe").is_some();
    let Analysis {
        layers,
        min_speedup,
        ..
    } = analyse_layers(args)?;
    outln!();
    outln!(
        "{:>6} {:>9} {:>12} {:>12} {:>9}  verdict",
        "layer",
        "experts",
        "req/token",
        "after",
        "speedup"
    );
    let mut worth = 0;
    for (l, r) in &layers {
        let v = if r.worth_repacking(min_speedup) {
            "repack"
        } else {
            "leave alone"
        };
        if r.worth_repacking(min_speedup) {
            worth += 1;
        }
        outln!(
            "{:>6} {:>9} {:>12.2} {:>12.2} {:>8.2}x  {}",
            l.block,
            l.n_experts,
            r.baseline.requests_per_token,
            r.optimized.requests_per_token,
            r.speedup(),
            v
        );
    }
    let mean: f64 = layers.iter().map(|(_, r)| r.speedup()).sum::<f64>() / layers.len() as f64;
    outln!("\nmean predicted speedup in expert read time: {mean:.2}x");
    outln!(
        "{worth} of {} layers clear the 1.05x threshold worth rewriting a file for",
        layers.len()
    );
    if !calibrated {
        outln!(
            "\nthese are request counts, not times — pass --probe pmx-probe.json for calibrated numbers"
        );
    }
    outln!(
        "\nreminder: this is the layout lever only. Raising I/O queue depth is worth several times\n\
         more on the same hardware, and requires an inference runtime rather than a repacked file."
    );
    Ok(())
}

fn cmd_plan(args: &Args) -> Result<(), String> {
    let out = args.get("out").unwrap_or("model.pmxplan").to_string();
    let ram_mib: u64 = args.num("ram-mib", 0)?;
    let model = args.req("model")?.to_string();
    let tracep = args.req("trace")?.to_string();
    let Analysis {
        trace: t,
        layers,
        min_speedup,
    } = analyse_layers(args)?;

    let mut plan = Plan {
        model,
        trace: tracep,
        layers: Vec::new(),
    };
    for (i, (l, r)) in layers.iter().enumerate() {
        if !r.worth_repacking(min_speedup) {
            continue;
        }
        plan.layers.push(LayerPlan {
            block: l.block,
            tensors: l.tensors.clone(),
            perm: r.layout.expert_at().to_vec(),
            predicted_speedup: r.speedup(),
            requests_before: r.baseline.requests_per_token,
            requests_after: r.optimized.requests_per_token,
        });
        // Report the allocation the same trace implies, even though this build
        // does not requantise.
        if ram_mib > 0 && i == 0 {
            let ca = CoActivation::from_trace(&t, i as u32);
            let cfg = PlanConfig {
                weights_per_expert: l.weights_per_expert,
                resident_budget_bytes: ram_mib << 20,
                baseline_bits: l.baseline_bits,
                tier_cost: TierCost::default(),
                ..PlanConfig::default()
            };
            let ap = pmx_plan::allocate(&ca, &Sensitivity::uniform(ca.n_experts as usize), &cfg);
            outln!(
                "\nallocation preview for block {} at a {} RAM budget:",
                l.block,
                human(ram_mib << 20)
            );
            outln!(
                "  checkpoint precision       {} ({:.2} bits/weight), {} per expert",
                l.baseline_type,
                l.baseline_bits,
                human(l.bytes_per_expert)
            );
            outln!(
                "  resident experts   {}",
                ap.experts
                    .iter()
                    .filter(|a| a.tier == pmx_plan::Tier::Resident)
                    .count()
            );
            outln!(
                "  selections served from RAM {:.1}%",
                pmx_plan::hit_rate(&ap) * 100.0
            );
            outln!("  predicted movement speedup {:.2}x", ap.speedup());
            outln!("  proxy loss ratio           {:.4}", ap.loss_ratio());
            outln!(
                "  NOTE: requantisation is not implemented in this build. These are the\n        \
                 allocation the trace implies and a proxy loss, not a measured result."
            );
        }
    }
    if plan.layers.is_empty() {
        outln!(
            "\nno layer clears the {min_speedup:.2}x threshold. Nothing worth repacking; no plan written.\n\
             Lower it with --min-speedup if you want a plan anyway."
        );
        return Ok(());
    }
    plan.write(&out)
        .map_err(|e| format!("writing {out}: {e}"))?;
    outln!("\nwrote {out} covering {} layers", plan.layers.len());
    outln!("review it, then:  potatomaxx pack --model <m> --plan {out} --out <out.gguf>");
    Ok(())
}

fn build_placement(g: &Gguf, plan: &Plan) -> Result<Placement, String> {
    let known: HashSet<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut perms = Vec::new();
    for l in &plan.layers {
        for name in &l.tensors {
            if !known.contains(name.as_str()) {
                return Err(format!("plan names tensor {name:?}, absent from the model"));
            }
            perms.push(PermSpec {
                tensor: name.clone(),
                perm: l.perm.clone(),
            });
        }
    }
    Ok(Placement {
        order: Vec::new(),
        group_align: 0,
        group_starts: HashSet::new(),
        perms,
    })
}

fn cmd_pack(args: &Args) -> Result<(), String> {
    let model = args.req("model")?;
    let planp = args.req("plan")?;
    let out = args.req("out")?;
    let g = open_model(model)?;
    let text = std::fs::read_to_string(planp).map_err(|e| format!("reading {planp}: {e}"))?;
    let plan = Plan::parse(&text).map_err(|e| format!("{planp}: {e}"))?;
    let placement = build_placement(&g, &plan)?;

    outln!("packing {model} -> {out}");
    outln!(
        "  {} layers, {} tensors permuted",
        plan.layers.len(),
        placement.perms.len()
    );
    let rep = pmx_gguf::repack(&g, &placement, out).map_err(|e| format!("repack failed: {e}"))?;
    outln!(
        "  wrote {} of tensor data ({} padding), header {}, data at {}",
        human(rep.data_bytes),
        human(rep.padding_bytes),
        human(rep.header_bytes),
        rep.data_start
    );
    outln!("\nnow prove it:  potatomaxx verify --model {model} --repacked {out} --plan {planp}");
    Ok(())
}

fn cmd_verify(args: &Args) -> Result<(), String> {
    let model = args.req("model")?;
    let repacked = args.req("repacked")?;
    let planp = args.req("plan")?;
    let a = open_model(model)?;
    let b = open_model(repacked)?;
    let text = std::fs::read_to_string(planp).map_err(|e| format!("reading {planp}: {e}"))?;
    let plan = Plan::parse(&text).map_err(|e| format!("{planp}: {e}"))?;
    let placement = build_placement(&a, &plan)?;

    outln!("verifying {repacked} against {model}");
    let rep = pmx_gguf::verify(&a, &b, &placement.perms)
        .map_err(|e| format!("VERIFICATION FAILED: {e}"))?;
    outln!(
        "  {} tensors compared, {} byte-identical, {} matched as a permutation",
        rep.tensors,
        rep.identical,
        rep.permuted
    );
    outln!("  {} of weights confirmed unchanged", human(rep.bytes));
    outln!("\nOK — the repacked file holds exactly the original weights.");
    Ok(())
}

#[cfg(test)]
mod fit_tests {
    use super::{fit_ceiling, fit_reserve, quant_weight_bytes, recommend_quant};

    const GIB: u64 = 1 << 30;

    #[test]
    fn reserve_grows_with_context() {
        assert!(fit_reserve(8192) > fit_reserve(1024));
    }

    #[test]
    fn eleven_gib_model_does_not_fit_eight_gib_ram() {
        let data = 10_480 * (1u64 << 20); // ~10.48 GiB weights
        let projected = data + fit_reserve(4096);
        let ram = (7.76 * GIB as f64) as u64;
        assert!(
            projected > fit_ceiling(ram),
            "should refuse to fit on 8 GiB"
        );
    }

    #[test]
    fn same_model_fits_sixteen_gib() {
        let data = 10_480 * (1u64 << 20);
        let projected = data + fit_reserve(4096);
        assert!(projected <= fit_ceiling(16 * GIB), "should fit on 16 GiB");
    }

    #[test]
    fn recommended_max_weights_is_below_ram() {
        let ram = (7.76 * GIB as f64) as u64;
        let max_weights = fit_ceiling(ram).saturating_sub(fit_reserve(4096));
        assert!(
            max_weights > 5 * GIB && max_weights < 7 * GIB,
            "≈6.5 GiB budget"
        );
    }

    #[test]
    fn recommendation_never_exceeds_budget() {
        let params: u64 = 8_000_000_000;
        let budget = 5 * GIB;
        let (_, sz) = recommend_quant(params, budget).expect("a small quant of an 8B fits 5 GiB");
        assert!(sz <= budget, "recommended size must fit the budget");
    }

    #[test]
    fn more_ram_never_recommends_a_smaller_tier() {
        let params: u64 = 8_000_000_000;
        let small = recommend_quant(params, 5 * GIB);
        let big = recommend_quant(params, 30 * GIB);
        // both fit for an 8B; the larger budget must not pick fewer bytes
        assert!(big.unwrap().1 >= small.unwrap().1);
    }

    #[test]
    fn budget_below_the_smallest_tier_returns_none() {
        // 30.5B params can't fit ~2 GiB even at 1-bit
        assert!(recommend_quant(30_500_000_000, 2 * GIB).is_none());
    }

    #[test]
    fn quant_size_scales_with_bpw() {
        let p = 1_000_000_000;
        assert!(quant_weight_bytes(p, 4.0) > quant_weight_bytes(p, 2.0));
    }
}
