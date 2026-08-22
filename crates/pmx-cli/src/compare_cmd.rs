// SPDX-License-Identifier: GPL-2.0-or-later
//! `potatomaxx compare` — check two quantisations of the same model agree.
//!
//! # Why this exists
//!
//! The dequantisers in `pmx-kernels` were written from the ggml block layouts
//! and tested against synthetic blocks constructed by hand. That proves the
//! arithmetic matches *my reading* of the layout, which is exactly the thing a
//! self-written test cannot check.
//!
//! Dequantising the same tensor from two different quantisations of the same
//! model is a real differential test. An F16 file is ground truth — half to float
//! is exhaustively verified over all 65536 bit patterns — so comparing a Q4_K
//! file against it validates the Q4_K decoder against something I did not write.
//! Two decoders being wrong in the same direction is implausible: Q4_K packs
//! 6-bit scales and mins with an irregular split across bytes, while Q6_K uses
//! 8-bit scales and a -32 bias. They share no code.
//!
//! It is also useful on its own: after requantising a model, this is how you
//! check what precision actually cost you, per tensor, on real weights rather
//! than on an analytic proxy.

use pmx_gguf::Gguf;
use pmx_kernels::{can_dequantize, ggml_dequant};

/// Agreement between one tensor in two files.
struct TensorDiff {
    name: String,
    ty_a: &'static str,
    ty_b: &'static str,
    n: usize,
    rmse: f64,
    max_abs: f64,
    /// RMSE relative to the RMS magnitude of the reference tensor.
    rel: f64,
    /// Pearson correlation. Near 1.0 means the same tensor; a decoder bug
    /// destroys this even when the error magnitude looks superficially small.
    corr: f64,
    nonfinite: usize,
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64, f64, usize) {
    let mut se = 0.0f64;
    let mut max = 0.0f64;
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sab = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut n = 0usize;
    let mut bad = 0usize;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            bad += 1;
            continue;
        }
        let (x, y) = (f64::from(*x), f64::from(*y));
        let d = x - y;
        se += d * d;
        if d.abs() > max {
            max = d.abs();
        }
        ref_sq += x * x;
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
        n += 1;
    }
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, bad);
    }
    let nf = n as f64;
    let rmse = (se / nf).sqrt();
    let rms_ref = (ref_sq / nf).sqrt();
    let rel = if rms_ref > 0.0 {
        rmse / rms_ref
    } else {
        f64::NAN
    };
    let num = sab / nf - (sa / nf) * (sb / nf);
    let den = ((saa / nf - (sa / nf).powi(2)) * (sbb / nf - (sb / nf).powi(2))).sqrt();
    let corr = if den > 0.0 { num / den } else { f64::NAN };
    (rmse, max, rel, corr, bad)
}

/// Compare tensors common to both files.
///
/// `a` is treated as the reference. `filter` restricts to tensor names
/// containing it; `limit` caps how many are compared, since a full model is a
/// lot of dequantisation.
pub fn run(path_a: &str, path_b: &str, filter: Option<&str>, limit: usize) -> Result<(), String> {
    let a = Gguf::open(path_a).map_err(|e| format!("reading {path_a}: {e}"))?;
    let b = Gguf::open(path_b).map_err(|e| format!("reading {path_b}: {e}"))?;

    println!("reference: {path_a}");
    println!(
        "           {} tensors, {}",
        a.tensors.len(),
        a.architecture().unwrap_or("?")
    );
    println!("candidate: {path_b}");
    println!(
        "           {} tensors, {}",
        b.tensors.len(),
        b.architecture().unwrap_or("?")
    );

    let by_name: std::collections::HashMap<&str, _> =
        b.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut rows: Vec<TensorDiff> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut da = Vec::new();
    let mut db = Vec::new();

    for ta in &a.tensors {
        if rows.len() >= limit {
            break;
        }
        if let Some(f) = filter {
            if !ta.name.contains(f) {
                continue;
            }
        }
        let tb = match by_name.get(ta.name.as_str()) {
            Some(t) => *t,
            None => {
                skipped.push(format!("{}: absent from candidate", ta.name));
                continue;
            }
        };
        if ta.dims != tb.dims {
            skipped.push(format!("{}: dims {:?} vs {:?}", ta.name, ta.dims, tb.dims));
            continue;
        }
        if !can_dequantize(ta.ggml_type.0) || !can_dequantize(tb.ggml_type.0) {
            skipped.push(format!(
                "{}: {} / {} not decodable by this build",
                ta.name,
                ta.ggml_type.name(),
                tb.ggml_type.name()
            ));
            continue;
        }
        let n = ta.n_elements().map_err(|e| e.to_string())? as usize;
        let raw_a = a.read_tensor(ta).map_err(|e| format!("{}: {e}", ta.name))?;
        let raw_b = b.read_tensor(tb).map_err(|e| format!("{}: {e}", tb.name))?;
        ggml_dequant::dequantize(ta.ggml_type.0, &raw_a, n, &mut da)
            .map_err(|e| format!("{} (reference): {e}", ta.name))?;
        ggml_dequant::dequantize(tb.ggml_type.0, &raw_b, n, &mut db)
            .map_err(|e| format!("{} (candidate): {e}", tb.name))?;

        let (rmse, max_abs, rel, corr, nonfinite) = stats(&da, &db);
        rows.push(TensorDiff {
            name: ta.name.clone(),
            ty_a: ta.ggml_type.name(),
            ty_b: tb.ggml_type.name(),
            n,
            rmse,
            max_abs,
            rel,
            corr,
            nonfinite,
        });
    }

    if rows.is_empty() {
        let mut m = String::from("no comparable tensors found");
        for s in skipped.iter().take(5) {
            m.push_str(&format!("\n  {s}"));
        }
        return Err(m);
    }

    println!(
        "\n{:<34} {:>7} {:>7} {:>11} {:>10} {:>9} {:>9}  {:>8}",
        "tensor", "ref", "cand", "elements", "rmse", "max.abs", "corr", "rel.err"
    );
    for r in &rows {
        println!(
            "{:<34} {:>7} {:>7} {:>11} {:>10.6} {:>9.5} {:>9.6}  {:>7.4}%",
            if r.name.len() > 34 {
                &r.name[r.name.len() - 34..]
            } else {
                &r.name
            },
            r.ty_a,
            r.ty_b,
            r.n,
            r.rmse,
            r.max_abs,
            r.corr,
            r.rel * 100.0
        );
        if r.nonfinite > 0 {
            println!("  ^ {} non-finite values skipped", r.nonfinite);
        }
    }

    // A decoder bug shows up as a collapsed correlation, not as a slightly
    // larger error. Judge on that.
    let worst_corr = rows.iter().map(|r| r.corr).fold(f64::INFINITY, f64::min);
    let mean_rel = rows.iter().map(|r| r.rel).sum::<f64>() / rows.len() as f64;
    let worst_rel = rows.iter().map(|r| r.rel).fold(0.0, f64::max);

    println!("\n{} tensors compared", rows.len());
    println!("  mean relative error   {:.4}%", mean_rel * 100.0);
    println!("  worst relative error  {:.4}%", worst_rel * 100.0);
    println!("  worst correlation     {:.6}", worst_corr);

    if worst_corr > 0.99 {
        println!(
            "\nPASS. Correlation above 0.99 on every tensor means both decoders recovered\n\
             the same underlying weights; the residual is quantisation error, which is what\n\
             it should be. A layout misreading in either decoder would destroy correlation,\n\
             not merely enlarge the error."
        );
    } else {
        println!(
            "\nFAIL. Correlation of {worst_corr:.4} is too low to be quantisation error alone.\n\
             One of these decoders is misreading its block layout."
        );
        return Err("differential comparison failed".into());
    }
    if !skipped.is_empty() {
        println!("\nskipped {} tensor(s):", skipped.len());
        for s in skipped.iter().take(6) {
            println!("  {s}");
        }
    }
    Ok(())
}
