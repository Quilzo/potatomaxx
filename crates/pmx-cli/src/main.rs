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

mod moe;
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
  probe        Measure this device's read bandwidth surface (blob size x queue depth)
  synth        Write a small synthetic MoE checkpoint and trace, for trying the pipeline
  inspect      Report the MoE structure of a GGUF checkpoint
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
           --locality <0..1>     share of tokens routed within one cluster (default 0.85)
           --no-scatter          keep planted clusters on contiguous expert ids
                                 (unrealistic: makes the existing order optimal)

  inspect  <model.gguf>

  analyse  --model <path> --trace <path> [--probe <path>] [--merge-gap <n>]
           [--queue-depth <n>]   reads the consuming runtime keeps in flight (default 8)
  plan     --model <path> --trace <path> [--probe <path>] [--merge-gap <n>]
           [--queue-depth <n>] [--ram-mib <n>] [--min-speedup <f>] [--out <path>]
           --min-speedup <f>     gain a layer must clear to be repacked (default 1.05)
  pack     --model <path> --plan <path> --out <path>
  verify   --model <path> --repacked <path> --plan <path>

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
        "analyse" | "analyze" => cmd_analyse(&args),
        "plan" => cmd_plan(&args),
        "pack" => cmd_pack(&args),
        "verify" => cmd_verify(&args),
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

    let mut t = Trace::synthetic(layers, experts, top_k, tokens, clusters, locality, 0xC0FFEE);
    // Planted clusters land on contiguous ids, which would make the checkpoint's
    // existing order already optimal. Real expert numbering has no such
    // locality, so scatter the labels unless asked not to.
    if args.get("no-scatter").is_none() {
        t.scatter_labels(0x5EED);
    }
    t.write(tracep)
        .map_err(|e| format!("writing {tracep}: {e}"))?;
    outln!(
        "wrote {tracep} — {tokens} tokens, top-{top_k}, {clusters} planted clusters, locality {locality}"
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
