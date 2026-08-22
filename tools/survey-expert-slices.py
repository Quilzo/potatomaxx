#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Survey expert slice sizes across real mixture-of-experts models.

This exists because it answers, cheaply and decisively, whether reordering
experts on disk can help any real checkpoint.

The measured bandwidth surface (`potatomaxx probe`) plateaus above roughly
256 KiB per request: below that, request size dominates throughput; above it,
queue depth is all that matters and coalescing reads buys nothing. So the layout
compiler only has a target if real models have expert slices under that size.

One expert contributes `hidden_size * intermediate_size` weights to each of its
three matrices, and each matrix is read as one slice. That is computable from a
model's config.json -- a couple of kilobytes -- instead of by downloading it.

Usage:  python3 tools/survey-expert-slices.py [--bpw 4.5] [repo ...]
"""
import argparse, json, sys, urllib.request

DEFAULT_MODELS = [
    "ibm-granite/granite-3.0-1b-a400m-instruct",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen3-30B-A3B",
    "deepseek-ai/DeepSeek-V2-Lite",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "microsoft/Phi-3.5-MoE-instruct",
]

# Where the measured surface stops rewarding larger requests.
BAND_HI = 256 * 1024


def fetch_config(repo, timeout=25):
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "potatomaxx-survey"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 - a survey should not die on one repo
        print(f"  {repo}: {exc}", file=sys.stderr)
        return None


def pick(cfg, *keys):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=float, default=4.5,
                    help="bits per weight of the quantisation (default 4.5, Q4_K)")
    ap.add_argument("repos", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()

    print(f"expert slice size at {args.bpw} bits/weight; "
          f"layout helps only below {BAND_HI // 1024} KiB\n")
    print(f"{'model':<44}{'experts':>8}{'hidden':>8}{'inter':>7}{'slice':>10}  verdict")
    print("-" * 94)

    rows = []
    for repo in args.repos:
        cfg = fetch_config(repo)
        if cfg is None:
            continue
        ne = pick(cfg, "num_local_experts", "num_experts", "n_routed_experts")
        hid = pick(cfg, "hidden_size", "d_model")
        inter = pick(cfg, "moe_intermediate_size", "intermediate_size", "ffn_dim")
        if None in (ne, hid, inter):
            print(f"{repo[:44]:<44}  fields missing "
                  f"(experts={ne} hidden={hid} inter={inter})")
            continue
        slice_bytes = hid * inter * args.bpw / 8
        verdict = "IN BAND" if slice_bytes < BAND_HI else "plateaued"
        print(f"{repo[:44]:<44}{ne:>8}{hid:>8}{inter:>7}"
              f"{slice_bytes / 1024:>8.0f} KiB  {verdict}")
        rows.append((repo, slice_bytes))

    if not rows:
        return 1
    in_band = [r for r in rows if r[1] < BAND_HI]
    smallest = min(rows, key=lambda r: r[1])
    print(f"\n{len(in_band)} of {len(rows)} models have slices under "
          f"{BAND_HI // 1024} KiB")
    print(f"smallest: {smallest[0]} at {smallest[1] / 1024:.0f} KiB")
    if not in_band:
        print("\nNo surveyed model is in the band where reordering experts can help.\n"
              "Layout is not the lever; see docs/REAL-MODEL-RESULTS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
