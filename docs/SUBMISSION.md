# Submitting potatomaxx to a foundation

Research into which foundations can actually accept this project, and the
prepared materials for the one that can. Written down rather than acted on,
because every route below requires the maintainer's identity and an ongoing
personal commitment.

## The options, and what blocks each

| Foundation | Licence | Verdict |
|---|---|---|
| **Apache Software Foundation** (Incubator) | ❌ Requires Apache-2.0 | **Ineligible.** ASF releases must be under Apache-2.0; a GPL-2.0 project does not qualify. Also needs a Champion who is an ASF Officer or Member, plus two or three mentors who are Incubator PMC members — a human relationship, not a form. |
| **Software Freedom Conservancy** | ✓ GPL is fine (OSI-approved and DFSG-free) | **Ineligible today.** Their published criteria state that projects "under one year old or proof-of-concept implementations are generally ineligible", and require "an existing, vibrant, diverse community". This project is days old with one contributor. Also: 10% of processed revenue to their general fund. **Revisit in a year with users.** |
| **LF AI & Data** (Sandbox) | ✓ OSI-approved is enough | **Blocked on membership.** Sandbox projects "must be contributed by an LF AI & Data member" — Quilzo would have to join LF AI & Data and the Linux Foundation, both paid. Also requires an OpenSSF Best Practices *Passing* badge, an executed Technical Charter and Project Contribution Agreement, and an affirmative TAC vote. |
| **CNCF** (Sandbox) | ✓ OSI-approved | **Wrong scope.** This is a local-inference storage tool, not cloud-native infrastructure. It would not survive the "is this cloud native?" question. |
| **GNU / FSF** | ✓ GPL is the entire point | **Viable now.** The only one that says so explicitly: *"If you can't answer all the questions or if the program does not fulfill all of the items mentioned, don't worry — that does not mean the package will be rejected. It's common for a program to be evaluated when it's not quite ready."* |

### Why GNU is the honest answer

Three things line up:

1. **Licence.** GNU asks for GPLv3-or-later. This project is GPL-2.0-**or-later**,
   which can be used under v3, so it satisfies GNU while remaining usable as v2 —
   the only choice that keeps both GNU and Linux-kernel compatibility open. Had we
   stayed on AGPL, or moved to GPLv3-only, one door or the other would have shut.
2. **Dependencies.** GNU requires that the software *and all its dependencies* be
   free software. This has **zero** dependencies beyond the Rust standard
   library, so the question is trivially satisfied — unusual, and worth stating.
3. **Maturity.** GNU is the only body here that evaluates programs which are not
   finished.

### What accepting GNU membership actually costs

Not a formality, and the maintainer should decide knowingly:

- **You become a GNU maintainer.** An ongoing obligation to stay in touch with the
  GNU Project, respond when they raise problems, and coordinate with other
  maintainers where packages interact.
- **You agree to follow GNU policies** — the coding standards, the maintainers
  document, release conventions.
- Copyright assignment to the FSF is **optional**. Keeping it means GPL
  enforcement stays your responsibility; assigning it means the FSF enforces.

## Gaps against GNU requirements

| Requirement | Status |
|---|---|
| Free software licence | ✓ GPL-2.0-or-later, SPDX on every file |
| All dependencies free | ✓ zero dependencies |
| GNU build conventions (`prefix`, `DESTDIR`, `install`, `uninstall`, `check`, `dist`) | ✓ `Makefile` added; verified staging-install and clean uninstall |
| Texinfo manual | ✓ `doc/potatomaxx.texi`, built in CI |
| Man page | ✓ `doc/potatomaxx.1` |
| Release tarball | ✓ `make dist` |
| Internationalisation via GNU Gettext | ✗ **not done.** All output is English. This is a "please" not a "must", and for a measurement tool whose output is numeric tables the value is arguable — but it is a real gap and should be discussed rather than hidden. |
| Similar-projects survey | ✓ below |
| Accessibility | Terminal output only; no curses, no colour-dependent meaning, no ANSI escapes. Pipes cleanly, which is the accessibility property that matters here. |

## Prepared questionnaire

To be emailed as plain text to `gnueval@gnu.org`. **Not sent** — it commits the
sender to GNU maintainership.

```
* General Information

** Do you agree to follow GNU policies?
   [MAINTAINER TO CONFIRM]

** Package name and version:
   potatomaxx 0.1.0

** Author Full Name <Email>:
   [MAINTAINER TO SUPPLY]

** URL to package home page:
   https://github.com/Quilzo/potatomaxx

** URL to source tarball:
   Produced by `make dist` (potatomaxx-0.1.0.tar.gz); also
   https://github.com/Quilzo/potatomaxx/archive/refs/tags/v0.1.0.tar.gz

** Brief description of the package:
   potatomaxx makes mixture-of-experts (MoE) language models cheaper to READ,
   so they can run on machines that cannot hold them in RAM.

   A large MoE checkpoint is streamed from storage while it runs, which makes
   decoding a storage problem. Storage cares enormously how you ask: on the
   development machine the same NVMe device delivers 0.099 GB/s at queue depth
   one and 3.291 GB/s at depth sixteen. potatomaxx reduces the cost of those
   reads three ways and measures each rather than asserting it: it reorders
   experts on disk so co-firing experts are adjacent; it stores each expert at
   its own bit width, chosen from measured quantisation error and observed
   access frequency; and it predicts which experts a layer will select before
   its router runs, so reads can be queued.

   The layout pass emits a drop-in GGUF -- same tensor names, same shapes,
   byte-identical weights -- because permuting the expert axis is a relabelling:
   permute the router's rows identically and the computed function is unchanged
   bit for bit. A `verify` subcommand proves this by comparing both files in
   full, and is strict enough to reject a single flipped byte.

   It is not an inference engine. It has no attention, no KV cache and no
   sampler. It makes a model cheaper to read; something else runs it.

* Code

** Dependencies:
   Source language: Rust (2021 edition, MSRV 1.75).
   Libraries: NONE. The package has zero dependencies beyond the Rust standard
   library. The GGUF container parser, the command-line parser, the io_uring
   implementation and the JSON reader are all written within the package. This
   is deliberate: the program parses model files downloaded from public hubs, so
   every dependency would widen the surface exposed to untrusted input.
   Build: cargo (part of the Rust toolchain), plus make and makeinfo.
   Optional at build time: groff to render the man page.

** Configuration, building, installation:
   A Makefile follows the GNU Makefile conventions: prefix, exec_prefix, bindir,
   datarootdir, mandir, infodir, docdir and DESTDIR are all honoured and
   overridable. Targets: all, build, check, lint, fmt, install, uninstall,
   installdirs, info, html, clean, distclean, dist. `make check` is the test
   target, as the standards require. Autoconf is not used, as there is nothing
   to probe: the toolchain handles platform differences, and kernel facilities
   are detected at run time rather than at configure time (distributions
   backport features, and containers misreport them).
   Verified: `make install DESTDIR=... prefix=/usr` stages correctly and
   `make uninstall` leaves no files behind.

** Documentation:
   Texinfo manual at doc/potatomaxx.texi, covering the problem, each
   subcommand, the correctness argument for repacking, the precision model, the
   kernel interfaces used, and the measurements which contradicted the design.
   It is built in continuous integration so it cannot silently rot. A man page
   is at doc/potatomaxx.1.

** Internationalization:
   Not yet done; all user-visible strings are English. The output is
   predominantly numeric tables and measured figures. We would welcome guidance
   on whether Gettext is expected for a tool of this shape, and are willing to
   do the work if so.

** Accessibility:
   Terminal output only. No curses interface, no colour, no ANSI escape
   sequences, and no information conveyed by colour or position alone. Output is
   plain text that pipes cleanly to other tools and to screen readers. Broken
   pipes are handled rather than treated as a crash.

** Security:
   No cryptography. No network access of any kind. No privileged operations, no
   setuid, no elevation.

   The security-relevant surface is that the program PARSES ATTACKER-CONTROLLED
   FILES: a GGUF checkpoint is downloaded from a public hub by someone who did
   not write it, and its header is read with offset arithmetic. That is the code
   path which, in other implementations of this format, produced CVE-2026-27940
   (an integer overflow producing an undersized heap allocation and then a
   528-byte controlled overflow, itself a bypass of the fix for CVE-2025-53630)
   and CVE-2026-7482, CVSS 9.1, a heap out-of-bounds read from inflated tensor
   dimensions that leaked process memory including credentials.

   The measures taken are therefore specific:
     - Every length, count, offset and dimension read from a file is validated
       with checked arithmetic against the real file size before it is trusted.
     - Every component forbids unsafe code except three, each for a reason safe
       Rust cannot express (SIMD intrinsics; page-aligned buffers for O_DIRECT;
       raw syscalls and shared io_uring mappings). In each the portable
       implementation is authoritative and the optimised path is tested against
       it.
     - A dedicated hostile-input suite runs on every change: 1400 random inputs,
       every single-byte mutation of a valid header, truncation at every length,
       inflated counts and dimensions, unaligned and out-of-range offsets, and
       invalid UTF-8. The contract is that ANY input yields success or an error,
       never a panic and never a read outside the file.
     - Tests run in debug as well as release builds, because debug enables the
       integer overflow checks release omits, and an overflow on a length taken
       from an attacker-controlled file is precisely the first CVE class above.
     - A reported vulnerability process is documented in SECURITY.md.

   The program writes files only where the user names them. It does not modify
   its input.

* Licensing:
   The package is GPL-2.0-or-later; every source file carries an SPDX
   identifier and the full text is in LICENSE. There are no dependencies, so
   there are no third-party licences to declare.

   On version: GPL-2.0-or-later was chosen so the same source can be used under
   GPLv3-or-later, satisfying GNU, while remaining usable as v2 by
   GPL-2.0-only projects such as the Linux kernel, whose interfaces this program
   uses and whose community it hopes to interoperate with. We are willing to
   discuss moving to GPL-3.0-or-later if GNU prefers, and would want to
   understand the consequence for kernel-adjacent reuse first.

   Documentation: the Texinfo manual is under GPL-2.0-or-later rather than the
   GFDL. We will relicense it to GFDL 1.3-or-later if that is required.

* Similar free software projects:

   Searched the Free Software Directory and the wider free-software ecosystem.
   Nothing in the Directory addresses this problem. The adjacent free software
   projects are:

   - llama.cpp (MIT) -- the dominant free local-inference engine. It reads GGUF,
     and pull request #25294 adds SSD-backed expert streaming. potatomaxx does
     not compete: its output is a drop-in GGUF, so llama.cpp reads a repacked
     file unchanged and benefits without modification.

   - Colibri (Apache-2.0) -- a mature C engine that streams MoE experts from
     disk with router-lookahead prefetch and learned pinning. Again complementary
     rather than competing: potatomaxx changes the byte layout and precision of
     the FILE, so an engine consumes the result.

   - MoE-Infinity -- a sparsity-aware expert cache for MoE serving.

   - Academic work on improving locality by changing ROUTING at training time
     (Oracle-MoE, Sticky Routing, ReMoE). potatomaxx modifies no weights, so it
     composes with all of them.

   The principal difference is one of layer. Every project above is a runtime or
   a training method. potatomaxx is neither: it is an offline transformation of
   the model file, plus the measurement tooling to decide whether that
   transformation is worth doing on a given machine. Nothing found reorders
   experts on disk by measured co-activation, and nothing allocates bit width
   per expert from observed access frequency and storage tier cost. The nearest
   precision work, APEX-Quant, allocates by structural role and layer position
   and uses no runtime traces.

   What motivated writing it: a 7.6 GB laptop with no GPU cannot hold a 30B MoE
   model, and the existing answers either need far more RAM or collapse to
   fractions of a token per second. Investigating why produced the finding the
   program is built around -- that queue depth is worth thirty-three times on
   such a machine, and that almost every engine leaves it on the table.

* Any other information, comments, or questions:

   Two things we would rather state than have discovered.

   FIRST, this program is young. It runs end to end, it has 162 tests covering
   correctness and hostile input, and continuous integration checks Linux and
   macOS. But it has been exercised on synthetic checkpoints and one laptop; it
   has not yet been run against a real multi-gigabyte MoE model. We are offering
   it for evaluation partly to find out whether GNU considers that premature.

   SECOND, the repository deliberately records the measurements that
   CONTRADICTED its own design, including the ones that weakened its central
   claim. Layout, which the program was originally built around, turned out to
   be the secondary lever. io_uring, which we implemented carefully, lost to a
   plain thread pool on the test machine. We keep these because a tool whose job
   is telling you whether an optimisation helps is worth nothing if it cannot be
   trusted when it says no. If that is not the house style, we would like to
   know early.

   We would also welcome a view on internationalisation (see above) and on
   whether the manual should move to the GFDL.
```

## If GNU is not the choice

Two other routes are worth more than a foundation right now, and neither needs
one:

- **NLnet / NGI Zero** run open grant calls for free-software infrastructure with
  no community-size requirement, which fits a young systems project better than
  any foundation does.
- **The relevant technical communities.** The `kio` findings — that queue depth is
  worth 33×, that batch-and-drain io_uring underperforms a thread pool, that
  `RWF_DONTCACHE` trades throughput for not evicting everyone else — are of direct
  interest to the `io-uring` and `linux-mm` lists, and cost nothing to share. Real
  users are what makes the SFC and LF routes possible in a year.
