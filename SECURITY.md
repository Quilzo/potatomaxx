# Security policy

## Reporting a vulnerability

Report privately via GitHub's **[Report a vulnerability](https://github.com/Quilzo/potatomaxx/security/advisories/new)**
form, which opens a private advisory visible only to the maintainers.

Please do **not** open a public issue for a security problem first.

Expect an acknowledgement within **7 days** and an assessment within **30 days**.
If a fix is warranted we will agree a disclosure date with you, and credit you
unless you ask otherwise.

## What counts as a vulnerability here

The threat model is narrow and specific, which makes it easy to be clear about.

**`potatomaxx` parses attacker-controlled files.** A GGUF checkpoint is downloaded
from a public hub by someone who did not write it, and this tool reads its header
with offset arithmetic. That is the same code path that produced, in other
implementations of this format:

- **CVE-2026-27940** — integer overflow in llama.cpp's `gguf_init_from_file_impl()`
  producing an undersized heap allocation, then a 528+ byte controlled overflow.
  Itself a bypass of the fix for CVE-2025-53630.
- **CVE-2026-7482** ("Bleeding Llama", CVSS 9.1) — out-of-bounds read from
  inflated tensor dimensions in Ollama's loader, leaking process memory.

So the following **are** vulnerabilities, and we want to hear about them:

- Any input to `Gguf::open`, a `.pmxstore`, a `.pmxtrace` or a `.pmxplan` that
  causes a **panic, abort, hang, unbounded allocation, or a read or write outside
  the file's real bounds**. The contract is that any input at all yields `Ok` or
  `Err`.
- Any way to make `potatomaxx verify` **accept a file whose weights differ** from
  the source. This is the project's central correctness claim; breaking it is the
  highest-severity bug we can have.
- Undefined behaviour in the three crates permitted `unsafe`: `pmx-kernels`
  (SIMD), `pmx-kio` (raw syscalls, shared io_uring mappings), `pmx-probe`
  (page-aligned buffers).
- A path traversal or unexpected write outside the output path a user named.

The following are **not** vulnerabilities:

- A malicious model producing bad *inference output*. This tool does not run
  inference, and cannot validate a model's semantics.
- Slow performance, or a plan whose predicted speedup does not materialise.
- Quality loss from an aggressive `--error-bits`. That is a documented trade the
  user chooses.
- Resource use proportional to an input a user deliberately supplied.

## Supported versions

Pre-1.0: only `main` is supported. There are no maintained release branches yet.

## How we try to prevent these

- Every crate is `#![forbid(unsafe_code)]` except the three named above, each with
  its invariants written down at the unsafe block.
- Every length, count, offset and dimension read from a file is validated with
  checked arithmetic against the real file size before it is trusted.
- A dedicated hostile-input suite (`cargo test -p pmx-gguf --test robustness`)
  runs on every push: 1,400 random inputs, every single-byte mutation of a valid
  header, truncation at every length, inflated counts and dimensions, unaligned
  and out-of-range offsets, invalid UTF-8.
- CI asserts that `verify` **rejects** a file with four bytes flipped.
- Tests run in debug as well as release, because debug enables the integer
  overflow checks release omits — and an overflow on a length from an
  attacker-controlled file is exactly bug class one above.
