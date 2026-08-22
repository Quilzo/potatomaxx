// SPDX-License-Identifier: GPL-2.0-or-later
//! Parser robustness against hostile input.
//!
//! Model files are downloaded from public hubs and parsed by hand-written offset
//! arithmetic. That code path is where this format's CVEs live: CVE-2026-27940
//! (integer overflow to undersized heap allocation, then a controlled overflow,
//! itself a bypass of the CVE-2025-53630 fix) and CVE-2026-7482 (out-of-bounds
//! read from inflated tensor dimensions, leaking process memory).
//!
//! The contract this suite enforces is narrow and absolute: **for any input at
//! all, `Gguf::open` returns `Ok` or `Err` — never a panic, never a hang, never a
//! read outside the file.** Rust turns the two CVE classes above into a checked
//! arithmetic error and a bounds check respectively, but only if the code
//! actually uses checked arithmetic and validates against the real file size, and
//! a test is the only way to know it does.
//!
//! This is a deterministic mutation sweep rather than a coverage-guided fuzzer.
//! It is not a substitute for `cargo fuzz`; it is what can run in CI on every
//! push with no extra tooling.

use pmx_gguf::Gguf;
use std::io::Write;
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join("pmx-gguf-robustness");
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// Deterministic xorshift, so a failure is always reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// A minimal but genuinely valid GGUF: one metadata key, one F32 tensor.
fn valid_gguf() -> Vec<u8> {
    const ALIGN: usize = 32;
    let dims: [u64; 2] = [64, 4];
    let n_elem: u64 = dims.iter().product();
    let nbytes = (n_elem * 4) as usize;

    let mut kv = Vec::new();
    put_str(&mut kv, "general.architecture");
    kv.extend_from_slice(&8u32.to_le_bytes()); // string
    put_str(&mut kv, "robustness");

    let mut h = Vec::new();
    h.extend_from_slice(b"GGUF");
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    h.extend_from_slice(&1u64.to_le_bytes()); // kv count
    h.extend_from_slice(&kv);
    put_str(&mut h, "t.weight");
    h.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        h.extend_from_slice(&d.to_le_bytes());
    }
    h.extend_from_slice(&0u32.to_le_bytes()); // F32
    h.extend_from_slice(&0u64.to_le_bytes()); // offset

    let pad = (ALIGN - h.len() % ALIGN) % ALIGN;
    h.resize(h.len() + pad, 0u8);
    h.resize(h.len() + nbytes, 0xABu8);
    h
}

fn parses(bytes: &[u8], name: &str) -> bool {
    let p = scratch(name);
    std::fs::write(&p, bytes).unwrap();
    let ok = Gguf::open(&p).is_ok();
    let _ = std::fs::remove_file(&p);
    ok
}

#[test]
fn the_baseline_file_is_actually_valid() {
    // If this fails, every negative result below is meaningless.
    let p = scratch("valid.gguf");
    std::fs::write(&p, valid_gguf()).unwrap();
    let g = Gguf::open(&p).expect("the handcrafted baseline must parse");
    assert_eq!(g.tensors.len(), 1);
    assert_eq!(g.tensors[0].name, "t.weight");
    assert_eq!(g.tensors[0].nbytes, 64 * 4 * 4);
    assert_eq!(g.architecture(), Some("robustness"));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn random_bytes_never_panic() {
    let mut rng = Rng(0x5EED_1234);
    for round in 0..600 {
        let len = (rng.next() % 4096) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        // Must not panic. Almost all of these are rejected at the magic.
        let _ = parses(&bytes, &format!("rand{round}.gguf"));
    }
}

#[test]
fn random_bytes_behind_a_valid_magic_never_panic() {
    // Far more interesting than pure noise: passing the magic check means the
    // parser proceeds into counts, lengths and offsets with garbage.
    let mut rng = Rng(0xC0FF_EE99);
    for round in 0..800 {
        let len = 8 + (rng.next() % 2048) as usize;
        let mut bytes = Vec::with_capacity(len);
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        while bytes.len() < len {
            bytes.push(rng.byte());
        }
        let _ = parses(&bytes, &format!("magic{round}.gguf"));
    }
}

#[test]
fn single_byte_mutations_of_a_valid_file_never_panic() {
    // Sweeps every byte of the header region, which is where all the arithmetic
    // lives. A mutation may legitimately still parse; it may never panic.
    let base = valid_gguf();
    let header_region = base.len().min(256);
    for pos in 0..header_region {
        for delta in [1u8, 0x7F, 0xFF] {
            let mut m = base.clone();
            m[pos] = m[pos].wrapping_add(delta);
            let _ = parses(&m, "mut1.gguf");
        }
    }
}

#[test]
fn inflated_counts_and_dimensions_are_rejected_without_allocating() {
    // The shape of CVE-2026-7482: enormous declared dimensions. The parser must
    // refuse rather than compute a huge length and read against it. If this ever
    // hangs or OOMs instead of returning, that is the bug.
    let base = valid_gguf();

    // tensor_count sits at offset 8, kv_count at 16.
    for (off, label) in [(8usize, "tensor_count"), (16, "kv_count")] {
        for v in [u64::MAX, u64::MAX / 2, 1 << 40, 1 << 32] {
            let mut m = base.clone();
            m[off..off + 8].copy_from_slice(&v.to_le_bytes());
            assert!(
                !parses(&m, "inflated_count.gguf"),
                "{label} = {v} must be rejected"
            );
        }
    }

    // The tensor's first dimension. Locate it by searching for the name.
    let needle = b"t.weight";
    let at = base
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("name present");
    // name bytes, then n_dims:u32, then dims.
    let dims_at = at + needle.len() + 4;
    for v in [u64::MAX, u64::MAX / 4, 1 << 50] {
        let mut m = base.clone();
        m[dims_at..dims_at + 8].copy_from_slice(&v.to_le_bytes());
        assert!(
            !parses(&m, "inflated_dim.gguf"),
            "a dimension of {v} must be rejected"
        );
    }
}

#[test]
fn truncation_at_every_length_never_panics_and_never_over_reads() {
    let base = valid_gguf();
    for cut in 0..base.len() {
        let p = scratch("trunc.gguf");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&base[..cut]).unwrap();
        drop(f);
        // A truncated file may legitimately still parse; if it does, every
        // tensor it claims must lie inside the file that actually exists.
        if let Ok(g) = Gguf::open(&p) {
            let len = std::fs::metadata(&p).unwrap().len();
            for t in &g.tensors {
                assert!(
                    g.data_start + t.offset + t.nbytes <= len,
                    "cut {cut}: tensor {:?} extends past the {len}-byte file",
                    t.name
                );
            }
        }
        let _ = std::fs::remove_file(&p);
    }
}

#[test]
fn an_offset_pointing_outside_the_data_section_is_rejected() {
    let base = valid_gguf();
    // The tensor offset is the last u64 of the tensor_info record.
    let needle = b"t.weight";
    let at = base
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    let off_at = at + needle.len() + 4 + 16 + 4; // n_dims, 2 dims, type
    for v in [u64::MAX, 1 << 40, 1 << 20] {
        let mut m = base.clone();
        m[off_at..off_at + 8].copy_from_slice(&v.to_le_bytes());
        assert!(
            !parses(&m, "badoff.gguf"),
            "tensor offset {v} lies outside the data section and must be rejected"
        );
    }
}

#[test]
fn a_non_utf8_tensor_name_is_rejected_cleanly() {
    let base = valid_gguf();
    let needle = b"t.weight";
    let at = base
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    let mut m = base.clone();
    m[at] = 0xFF; // invalid UTF-8 lead byte
    m[at + 1] = 0xFE;
    assert!(
        !parses(&m, "badutf8.gguf"),
        "invalid UTF-8 must be rejected"
    );
}

#[test]
fn an_unaligned_tensor_offset_is_rejected() {
    let base = valid_gguf();
    let needle = b"t.weight";
    let at = base
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    let off_at = at + needle.len() + 4 + 16 + 4;
    let mut m = base.clone();
    // 7 is not a multiple of the 32-byte default alignment.
    m[off_at..off_at + 8].copy_from_slice(&7u64.to_le_bytes());
    assert!(
        !parses(&m, "unaligned.gguf"),
        "an offset violating general.alignment must be rejected"
    );
}
