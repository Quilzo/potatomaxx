//! Building a small synthetic MoE checkpoint and a matching routing trace.
//!
//! The whole pipeline — detect, analyse, plan, pack, verify — should be
//! runnable in a couple of seconds without downloading tens of gigabytes. Each
//! expert slice is filled with a pattern derived from its layer, tensor and
//! expert index, so a permutation that scrambles experts (rather than
//! relabelling them consistently) shows up immediately in `verify`.

use std::io::{BufWriter, Write};
use std::path::Path;

/// Shape of the synthetic model.
#[derive(Debug, Clone, Copy)]
pub struct SynthShape {
    /// MoE layers.
    pub layers: u32,
    /// Experts per layer.
    pub experts: u32,
    /// Input width.
    pub d_in: u64,
    /// Hidden width of one expert.
    pub d_ff: u64,
}

impl Default for SynthShape {
    fn default() -> Self {
        SynthShape {
            layers: 2,
            experts: 32,
            d_in: 64,
            d_ff: 128,
        }
    }
}

const F16: u32 = 1;
const F32: u32 = 0;
const ALIGN: u64 = 32;

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// Deterministic byte pattern for one expert slice.
fn pattern(layer: u32, tensor: u32, expert: u32, len: usize, out: &mut Vec<u8>) {
    out.clear();
    let mut x = ((layer as u64) << 40) ^ ((tensor as u64) << 24) ^ (expert as u64 * 0x9E3779B9) | 1;
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(len);
}

/// Write a synthetic MoE GGUF to `path`. Returns the byte length written.
pub fn write_gguf(path: impl AsRef<Path>, shape: SynthShape) -> std::io::Result<u64> {
    struct T {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        nbytes: u64,
        layer: u32,
        kind: u32,
    }
    let mut tensors: Vec<T> = Vec::new();
    for l in 0..shape.layers {
        for (kind, base) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .enumerate()
        {
            let n = shape.d_in * shape.d_ff * u64::from(shape.experts);
            tensors.push(T {
                name: format!("blk.{l}.{base}.weight"),
                dims: vec![shape.d_in, shape.d_ff, u64::from(shape.experts)],
                ty: F16,
                nbytes: n * 2,
                layer: l,
                kind: kind as u32,
            });
        }
        let n = shape.d_in * u64::from(shape.experts);
        tensors.push(T {
            name: format!("blk.{l}.ffn_gate_inp.weight"),
            dims: vec![shape.d_in, u64::from(shape.experts)],
            ty: F32,
            nbytes: n * 4,
            layer: l,
            kind: 3,
        });
    }
    // A non-expert tensor, so layouts have something to leave alone.
    tensors.push(T {
        name: "token_embd.weight".to_string(),
        dims: vec![shape.d_in, 256],
        ty: F16,
        nbytes: shape.d_in * 256 * 2,
        layer: u32::MAX,
        kind: 4,
    });

    // Metadata.
    let mut kv: Vec<u8> = Vec::new();
    let mut n_kv = 0u64;
    put_str(&mut kv, "general.architecture");
    kv.extend_from_slice(&8u32.to_le_bytes()); // string
    put_str(&mut kv, "pmxsynth");
    n_kv += 1;
    put_str(&mut kv, "general.alignment");
    kv.extend_from_slice(&4u32.to_le_bytes()); // uint32
    kv.extend_from_slice(&(ALIGN as u32).to_le_bytes());
    n_kv += 1;
    put_str(&mut kv, "pmxsynth.expert_count");
    kv.extend_from_slice(&4u32.to_le_bytes());
    kv.extend_from_slice(&shape.experts.to_le_bytes());
    n_kv += 1;

    // Offsets, laid out in declaration order with alignment padding.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for t in &tensors {
        let pad = (ALIGN - cur % ALIGN) % ALIGN;
        cur += pad;
        offsets.push(cur);
        cur += t.nbytes;
    }

    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&n_kv.to_le_bytes());
    header.extend_from_slice(&kv);
    for (t, off) in tensors.iter().zip(&offsets) {
        put_str(&mut header, &t.name);
        header.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            header.extend_from_slice(&d.to_le_bytes());
        }
        header.extend_from_slice(&t.ty.to_le_bytes());
        header.extend_from_slice(&off.to_le_bytes());
    }
    let data_start = {
        let h = header.len() as u64;
        h + (ALIGN - h % ALIGN) % ALIGN
    };

    let mut w = BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&header)?;
    w.write_all(&vec![0u8; (data_start - header.len() as u64) as usize])?;

    let mut written = 0u64;
    let mut buf = Vec::new();
    for (t, off) in tensors.iter().zip(&offsets) {
        while written < *off {
            let pad = (*off - written).min(4096) as usize;
            w.write_all(&vec![0u8; pad])?;
            written += pad as u64;
        }
        if t.layer == u32::MAX {
            pattern(0, t.kind, 0, t.nbytes as usize, &mut buf);
            w.write_all(&buf)?;
        } else {
            let slices = *t.dims.last().unwrap();
            let per = (t.nbytes / slices) as usize;
            for e in 0..slices as u32 {
                pattern(t.layer, t.kind, e, per, &mut buf);
                w.write_all(&buf)?;
            }
        }
        written += t.nbytes;
    }
    w.flush()?;
    Ok(data_start + written)
}
