# Governance

`potatomaxx` is early and small, and this document says so plainly rather than
describing a structure that does not exist yet.

## Current state

The project was started in 2026 by **[rsh1k](https://github.com/rsh1k)**
(Rashik Adhikari) under the [Quilzo](https://github.com/Quilzo) organisation, and
currently has a single maintainer. See [AUTHORS](AUTHORS). Decisions are made in public on the issue
tracker and in pull requests.

## Decision making

- **Technical changes** go through a pull request. Anyone may open one. A
  maintainer reviews and merges.
- **Disagreements** are resolved by discussion in the open, on the relevant issue
  or pull request. Where consensus is not reached, the maintainer decides and
  records the reasoning in the thread.
- **Anything that changes a measured claim** — a benchmark number, a stated
  speedup, a correctness guarantee — requires the measurement, on stated
  hardware, in the pull request. This is the one rule the project will not bend:
  the value of a tool that tells you whether an optimisation helps is entirely in
  being trustworthy when it says no.

## Becoming a maintainer

Anyone who has contributed substantively and shown good judgement about the point
above may be invited to become a maintainer by an existing maintainer. There is no
fixed threshold; sustained, careful contribution is the whole criterion.

The project would benefit from more maintainers, particularly anyone with real
MoE-serving hardware or kernel storage experience.

## Contribution requirements

See [CONTRIBUTING.md](CONTRIBUTING.md). In summary: `Signed-off-by:` per the
Developer's Certificate of Origin, an SPDX header on new files, no new
dependencies without a strong argument, and no `unsafe` outside the three crates
that already have it.

## Licence and relicensing

GPL-2.0-or-later. Contributors retain their copyright; there is no CLA and no
copyright assignment.

A consequence worth stating: because copyright is distributed across
contributors, **relicensing would require the agreement of every contributor**.
That is deliberate. It means the licence cannot be quietly changed later, which
is a guarantee to anyone contributing under it.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Reports go to the maintainers via
the address listed there.
