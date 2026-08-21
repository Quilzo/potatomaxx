//! Recognising MoE structure in a GGUF checkpoint.
//!
//! GGUF names expert tensors `blk.<n>.ffn_{gate,up,down}_exps.weight`, stacking
//! every expert of the layer into one tensor whose last axis is the expert
//! index, and names the router `blk.<n>.ffn_gate_inp.weight`. Some
//! architectures add a per-expert bias, `blk.<n>.exp_probs_b.bias`.
//!
//! Everything that carries the expert axis for a layer must be permuted
//! together, or the model silently computes something else. This module's job is
//! to find that complete set, and to refuse when it cannot.

use pmx_gguf::{Gguf, TensorInfo};
use std::collections::BTreeMap;

/// Tensor name fragments that carry the expert axis and must move together.
const EXPERT_TENSORS: &[&str] = &[
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_exps_b",
    "ffn_up_exps_b",
    "ffn_down_exps_b",
];
/// Router tensors, whose expert-indexed axis selects among the experts.
const ROUTER_TENSORS: &[&str] = &["ffn_gate_inp", "exp_probs_b"];

/// One MoE layer's tensors.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    /// Block index.
    pub block: u32,
    /// Number of experts, from the expert tensors' last axis.
    pub n_experts: u64,
    /// Names of tensors carrying the expert axis (experts and router alike).
    pub tensors: Vec<String>,
    /// Names of just the big expert-weight tensors.
    pub expert_tensors: Vec<String>,
    /// Bytes in one expert's slice of one expert tensor, summed over tensors.
    pub bytes_per_expert: u64,
    /// Bytes in one expert's slice of the largest expert tensor.
    pub slice_bytes: u64,
    /// Weights in one expert, summed over expert tensors.
    pub weights_per_expert: u64,
    /// Bits per weight the checkpoint actually stores, averaged over the expert
    /// tensors. Read from their GGML types rather than assumed.
    pub baseline_bits: f64,
    /// Label of the dominant expert tensor type, for reporting.
    pub baseline_type: &'static str,
}

/// What was found in a checkpoint.
#[derive(Debug, Clone, Default)]
pub struct MoeModel {
    /// Layers discovered, in block order.
    pub layers: Vec<MoeLayer>,
    /// Tensors that look expert-shaped but were rejected, with the reason.
    pub skipped: Vec<(String, String)>,
}

fn block_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (num, _) = rest.split_once('.')?;
    num.parse().ok()
}

fn matches_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| name.contains(n))
}

/// Find every MoE layer whose expert axis can be permuted safely.
pub fn detect(g: &Gguf) -> MoeModel {
    let mut per_block: BTreeMap<u32, Vec<&TensorInfo>> = BTreeMap::new();
    let mut out = MoeModel::default();

    for t in &g.tensors {
        if !matches_any(&t.name, EXPERT_TENSORS) && !matches_any(&t.name, ROUTER_TENSORS) {
            continue;
        }
        match block_of(&t.name) {
            Some(b) => per_block.entry(b).or_default().push(t),
            None => out.skipped.push((
                t.name.clone(),
                "no blk.<n> prefix; cannot attribute to a layer".into(),
            )),
        }
    }

    for (block, tensors) in per_block {
        let experts: Vec<&&TensorInfo> = tensors
            .iter()
            .filter(|t| matches_any(&t.name, EXPERT_TENSORS))
            .collect();
        if experts.is_empty() {
            continue;
        }
        // Every expert-axis tensor must agree on the expert count.
        let counts: Vec<u64> = tensors
            .iter()
            .map(|t| *t.dims.last().unwrap_or(&0))
            .collect();
        let n_experts = *counts.iter().max().unwrap_or(&0);
        let mut ok = true;
        let mut names = Vec::new();
        for t in &tensors {
            let last = *t.dims.last().unwrap_or(&0);
            if last != n_experts {
                out.skipped.push((
                    t.name.clone(),
                    format!("expert axis is {last}, but the layer has {n_experts} experts"),
                ));
                ok = false;
                continue;
            }
            if n_experts == 0 || t.nbytes % n_experts != 0 {
                out.skipped.push((
                    t.name.clone(),
                    format!(
                        "{} bytes does not divide into {n_experts} slices; quantisation blocks straddle the expert axis",
                        t.nbytes
                    ),
                ));
                ok = false;
                continue;
            }
            names.push(t.name.clone());
        }
        if !ok || names.is_empty() {
            continue;
        }

        let expert_names: Vec<String> = experts.iter().map(|t| t.name.clone()).collect();
        let bytes_per_expert: u64 = experts.iter().map(|t| t.nbytes / n_experts).sum();
        let slice_bytes = experts
            .iter()
            .map(|t| t.nbytes / n_experts)
            .max()
            .unwrap_or(0);
        let weights_per_expert: u64 = experts
            .iter()
            .map(|t| t.n_elements().unwrap_or(0) / n_experts.max(1))
            .sum();

        // Average bits per weight across the expert tensors, weighted by size.
        let mut bit_bytes = 0f64;
        let mut bit_weights = 0f64;
        let mut biggest = (0u64, "?");
        for t in &experts {
            let w = t.n_elements().unwrap_or(0) as f64;
            if w == 0.0 {
                continue;
            }
            bit_bytes += t.nbytes as f64 * 8.0;
            bit_weights += w;
            if t.nbytes > biggest.0 {
                biggest = (t.nbytes, t.ggml_type.name());
            }
        }
        let baseline_bits = if bit_weights > 0.0 {
            bit_bytes / bit_weights
        } else {
            0.0
        };

        out.layers.push(MoeLayer {
            block,
            n_experts,
            tensors: names,
            expert_tensors: expert_names,
            bytes_per_expert,
            slice_bytes,
            weights_per_expert,
            baseline_bits,
            baseline_type: biggest.1,
        });
    }
    out
}
