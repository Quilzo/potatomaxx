# Contributing

## The one rule

**A repacked model must hold exactly the original weights.** `potatomaxx`
rewrites files that took a lot of compute to produce; silently corrupting one
would be the worst thing this tool could do.

So any change touching `pmx-gguf` or the permutation path must keep
`potatomaxx verify` passing, and must keep the tamper test passing — the one that
flips bytes in a repacked file and asserts verification *fails*. If you find a
way to make verification pass on a file it should reject, that is the highest
priority bug in the project.

## Ground rules

- **Zero dependencies.** The workspace is `std` only. A new dependency needs a
  strong argument; "it would be convenient" is not one. Untrusted model files are
  parsed here, and every dependency widens that surface.
- **`unsafe` stays quarantined.** Every crate is `#![forbid(unsafe_code)]` except
  `pmx-probe`, which needs page-aligned buffers for `O_DIRECT`. New `unsafe`
  outside that crate will be rejected. Inside it, each block carries a written
  invariant.
- **Never trust a number from a file.** Lengths, counts, offsets and dimensions
  all come from attacker-controlled input. Use checked arithmetic, validate
  against the real file size, and return a typed error. See the CVEs in the
  README for what happens otherwise.
- **Don't oversell a measurement.** If a figure is computed from a model rather
  than measured on hardware, say so where it is reported. The README has a table
  distinguishing the two; keep it accurate. A tool whose job is to tell you
  whether an optimisation helps has to be trusted when it says no.

## Running things

```bash
cargo test                 # 42 tests, no network, no fixtures needed
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# End-to-end, on synthetic data:
cargo run --release -- synth
cargo run --release -- probe --corpus-mib 512 --out surf.json
cargo run --release -- analyse --model synth.gguf --trace synth.pmxtrace --probe surf.json
```

`probe` writes a scratch corpus in the target directory and deletes it
afterwards. It touches nothing else.

## Things worth doing

- **Run it on a real MoE checkpoint** and report the `analyse` output —
  particularly if the answer is "not worth repacking". Real numbers on real
  files are the most useful contribution right now.
- **Requantisation kernels**, so `pmx-plan`'s allocation can actually be
  applied. This needs a quality evaluation alongside it; the allocator's
  `delta_loss` is currently a documented proxy, not a measurement.
- **Measured per-expert sensitivity** to replace that proxy.
- **Probe on other storage.** The bandwidth surface is device-specific, and every
  conclusion here follows from its shape. SATA SSDs, eMMC, spinning disks and
  Apple unified memory will each say something different.
