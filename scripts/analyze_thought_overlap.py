#!/usr/bin/env python3
"""Analyze overlap across Tree-of-Thought candidates to motivate prefix sharing.

This script parses ToT run logs (the JSON files produced by run.py), reconstructs
parent-child expansions at each step, and reports overlap metrics that are useful
for reasoning about prefix-sharing/KV-cache reuse potential.

Outputs:
1. step_metrics.csv: one row per (problem idx, step)
2. run_metrics.csv: one row per problem idx
3. summary.json: aggregate stats and top overlap steps
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


WORD_RE = re.compile(r"\S+")


def normalize_text(value: object) -> str:
    """Normalize log text fields for stable overlap metrics."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def tokenize(text: str, mode: str) -> List[str]:
    if mode == "char":
        return list(text)
    if mode == "word":
        return WORD_RE.findall(text)
    raise ValueError(f"Unsupported tokenizer mode: {mode}")


def lcp_len(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    n = min(len(seq_a), len(seq_b))
    i = 0
    while i < n and seq_a[i] == seq_b[i]:
        i += 1
    return i


def lcp_len_many(token_lists: Sequence[Sequence[str]]) -> int:
    if not token_lists:
        return 0
    prefix = list(token_lists[0])
    for tokens in token_lists[1:]:
        prefix_len = lcp_len(prefix, tokens)
        if prefix_len == 0:
            return 0
        prefix = prefix[:prefix_len]
    return len(prefix)


def pair_from_linear_index(index: int, n_items: int) -> Tuple[int, int]:
    """Map a linear index in [0, nC2) to a unique pair (i, j), i < j."""
    remaining = index
    for i in range(n_items - 1):
        width = n_items - i - 1
        if remaining < width:
            return i, i + 1 + remaining
        remaining -= width
    raise ValueError("Pair index out of range")


@dataclass
class PairwiseStats:
    pair_count_total: int
    pair_count_used: int
    mean_lcp_tokens: float
    mean_lcp_ratio: float


def compute_pairwise_stats(
    tokenized_texts: Sequence[Sequence[str]],
    max_pairs: int,
    rng: random.Random,
) -> PairwiseStats:
    n = len(tokenized_texts)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return PairwiseStats(0, 0, 0.0, 0.0)

    if max_pairs <= 0 or total_pairs <= max_pairs:
        sampled_indices = range(total_pairs)
        used_pairs = total_pairs
    else:
        sampled_indices = rng.sample(range(total_pairs), max_pairs)
        used_pairs = max_pairs

    sum_lcp = 0.0
    sum_ratio = 0.0
    for linear_idx in sampled_indices:
        i, j = pair_from_linear_index(linear_idx, n)
        tokens_a = tokenized_texts[i]
        tokens_b = tokenized_texts[j]
        lcp = lcp_len(tokens_a, tokens_b)
        denom = min(len(tokens_a), len(tokens_b))
        ratio = (lcp / denom) if denom > 0 else 0.0
        sum_lcp += lcp
        sum_ratio += ratio

    return PairwiseStats(
        pair_count_total=total_pairs,
        pair_count_used=used_pairs,
        mean_lcp_tokens=(sum_lcp / used_pairs) if used_pairs else 0.0,
        mean_lcp_ratio=(sum_ratio / used_pairs) if used_pairs else 0.0,
    )


def assign_parents(candidates: Sequence[str], parents: Sequence[str]) -> Tuple[List[Optional[str]], int]:
    """Assign each candidate to the longest parent that is a string prefix."""
    sorted_parents = sorted((normalize_text(p) for p in parents), key=len, reverse=True)
    assignments: List[Optional[str]] = []
    misses = 0

    for candidate in candidates:
        assigned: Optional[str] = None
        for parent in sorted_parents:
            if candidate.startswith(parent):
                assigned = parent
                break
        if assigned is None:
            misses += 1
        assignments.append(assigned)
    return assignments, misses


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def load_log_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported top-level JSON type in {path}: {type(data).__name__}")


def expand_input_paths(inputs: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for item in inputs:
        if os.path.isfile(item):
            resolved.append(item)
            continue
        matches = glob.glob(item, recursive=True)
        if matches:
            resolved.extend(path for path in matches if os.path.isfile(path))
    return sorted(set(resolved))


def step_to_int(step: object, fallback: int) -> int:
    if isinstance(step, int):
        return step
    try:
        return int(step)
    except (TypeError, ValueError):
        return fallback


def run_analysis(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    rng = random.Random(args.seed)
    step_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    paths = expand_input_paths(args.inputs)
    if not paths:
        raise ValueError("No input files matched. Pass explicit files or glob patterns.")

    for path in paths:
        records = load_log_records(path)
        for record_id, record in enumerate(records):
            steps = record.get("steps", [])
            if not isinstance(steps, list) or not steps:
                continue

            idx_value = record.get("idx", record_id)
            run_step_rows: List[Dict[str, object]] = []

            for fallback_step, step_obj in enumerate(steps):
                if not isinstance(step_obj, dict):
                    continue

                step_id = step_to_int(step_obj.get("step"), fallback_step)
                parents = [normalize_text(item) for item in step_obj.get("ys", [])]
                candidates = [normalize_text(item) for item in step_obj.get("new_ys", [])]
                selected = [normalize_text(item) for item in step_obj.get("select_new_ys", [])]

                if len(candidates) < args.min_candidates:
                    continue

                candidate_token_lists = [tokenize(text, args.tokenizer) for text in candidates]
                candidate_token_lens = [len(tokens) for tokens in candidate_token_lists]

                assignments, assignment_misses = assign_parents(candidates, parents)
                children_by_parent: Dict[str, List[int]] = defaultdict(list)
                for child_idx, parent in enumerate(assignments):
                    if parent is not None:
                        children_by_parent[parent].append(child_idx)

                parent_token_len_cache = {
                    parent: len(tokenize(parent, args.tokenizer))
                    for parent in children_by_parent
                }

                total_candidate_tokens = sum(candidate_token_lens)
                total_parent_prefix_tokens = 0
                reusable_parent_prefix_tokens = 0
                reusable_group_prefix_tokens = 0
                branching_counts: List[int] = []

                for parent, child_indices in children_by_parent.items():
                    branch_size = len(child_indices)
                    if branch_size <= 0:
                        continue
                    branching_counts.append(branch_size)

                    parent_token_len = parent_token_len_cache[parent]
                    total_parent_prefix_tokens += branch_size * parent_token_len
                    if branch_size > 1:
                        reusable_parent_prefix_tokens += (branch_size - 1) * parent_token_len

                duplicate_rate = 1.0 - (len(set(candidates)) / len(candidates))
                pairwise = compute_pairwise_stats(candidate_token_lists, args.max_pairs, rng)

                # Sibling overlap: pairwise only within each parent group (suffix-only).
                sibling_pair_total = 0
                sibling_pair_used = 0
                sibling_lcp_sum = 0.0
                sibling_ratio_sum = 0.0
                sibling_shared_suffix_tokens_sum = 0
                for parent, child_indices in children_by_parent.items():
                    if len(child_indices) < 2:
                        continue
                    parent_char_len = len(parent)
                    suffixes: List[List[str]] = []
                    for child_idx in child_indices:
                        suffixes.append(tokenize(candidates[child_idx][parent_char_len:], args.tokenizer))

                    shared_suffix_tokens = lcp_len_many(suffixes)
                    sibling_shared_suffix_tokens_sum += shared_suffix_tokens
                    reusable_group_prefix_tokens += (len(child_indices) - 1) * (
                        len(tokenize(parent, args.tokenizer)) + shared_suffix_tokens
                    )

                    sibling_stats = compute_pairwise_stats(suffixes, args.max_pairs, rng)
                    sibling_pair_total += sibling_stats.pair_count_total
                    sibling_pair_used += sibling_stats.pair_count_used
                    sibling_lcp_sum += sibling_stats.mean_lcp_tokens * sibling_stats.pair_count_used
                    sibling_ratio_sum += sibling_stats.mean_lcp_ratio * sibling_stats.pair_count_used

                selected_token_lists = [tokenize(text, args.tokenizer) for text in selected]
                selected_pair_stats = compute_pairwise_stats(selected_token_lists, args.max_pairs, rng)

                parent_prefix_fraction = (
                    (total_parent_prefix_tokens / total_candidate_tokens)
                    if total_candidate_tokens > 0
                    else 0.0
                )
                reusable_vs_candidate = (
                    (reusable_parent_prefix_tokens / total_candidate_tokens)
                    if total_candidate_tokens > 0
                    else 0.0
                )
                reusable_vs_parent_prefix = (
                    (reusable_parent_prefix_tokens / total_parent_prefix_tokens)
                    if total_parent_prefix_tokens > 0
                    else 0.0
                )

                row: Dict[str, object] = {
                    "source_file": path,
                    "record_idx": idx_value,
                    "step": step_id,
                    "num_parents": len(parents),
                    "num_candidates": len(candidates),
                    "num_selected": len(selected),
                    "num_unique_candidates": len(set(candidates)),
                    "duplicate_rate": duplicate_rate,
                    "mean_candidate_tokens": safe_mean(candidate_token_lens),
                    "median_candidate_tokens": safe_median(candidate_token_lens),
                    "total_candidate_tokens": total_candidate_tokens,
                    "parent_assignment_misses": assignment_misses,
                    "parent_assignment_miss_rate": (
                        assignment_misses / len(candidates) if candidates else 0.0
                    ),
                    "num_parent_groups": len(children_by_parent),
                    "num_branching_groups": sum(1 for v in children_by_parent.values() if len(v) > 1),
                    "avg_children_per_group": safe_mean(branching_counts),
                    "max_children_in_group": max(branching_counts) if branching_counts else 0,
                    "total_parent_prefix_tokens": total_parent_prefix_tokens,
                    "reusable_parent_prefix_tokens": reusable_parent_prefix_tokens,
                    "reusable_group_prefix_tokens": reusable_group_prefix_tokens,
                    "sibling_shared_suffix_tokens": sibling_shared_suffix_tokens_sum,
                    "parent_prefix_fraction_of_candidate_tokens": parent_prefix_fraction,
                    "reusable_prefix_fraction_of_candidate_tokens": reusable_vs_candidate,
                    "reusable_fraction_of_parent_prefix_tokens": reusable_vs_parent_prefix,
                    "reusable_group_prefix_fraction_of_candidate_tokens": (
                        reusable_group_prefix_tokens / total_candidate_tokens if total_candidate_tokens > 0 else 0.0
                    ),
                    "pair_count_total": pairwise.pair_count_total,
                    "pair_count_used": pairwise.pair_count_used,
                    "pairwise_mean_lcp_tokens": pairwise.mean_lcp_tokens,
                    "pairwise_mean_lcp_ratio": pairwise.mean_lcp_ratio,
                    "sibling_pair_count_total": sibling_pair_total,
                    "sibling_pair_count_used": sibling_pair_used,
                    "sibling_suffix_mean_lcp_tokens": (
                        sibling_lcp_sum / sibling_pair_used if sibling_pair_used > 0 else 0.0
                    ),
                    "sibling_suffix_mean_lcp_ratio": (
                        sibling_ratio_sum / sibling_pair_used if sibling_pair_used > 0 else 0.0
                    ),
                    "selected_pair_count_total": selected_pair_stats.pair_count_total,
                    "selected_pair_count_used": selected_pair_stats.pair_count_used,
                    "selected_mean_lcp_tokens": selected_pair_stats.mean_lcp_tokens,
                    "selected_mean_lcp_ratio": selected_pair_stats.mean_lcp_ratio,
                }

                step_rows.append(row)
                run_step_rows.append(row)

            if run_step_rows:
                run_candidate_tokens = sum(float(row["total_candidate_tokens"]) for row in run_step_rows)
                run_parent_tokens = sum(float(row["total_parent_prefix_tokens"]) for row in run_step_rows)
                run_reusable_tokens = sum(float(row["reusable_parent_prefix_tokens"]) for row in run_step_rows)
                run_reusable_group_tokens = sum(float(row["reusable_group_prefix_tokens"]) for row in run_step_rows)
                run_pairs_used = sum(int(row["pair_count_used"]) for row in run_step_rows)
                run_pairwise_lcp_weighted = sum(
                    float(row["pairwise_mean_lcp_ratio"]) * int(row["pair_count_used"]) for row in run_step_rows
                )
                run_rows.append(
                    {
                        "source_file": path,
                        "record_idx": idx_value,
                        "steps_analyzed": len(run_step_rows),
                        "candidate_tokens": run_candidate_tokens,
                        "parent_prefix_tokens": run_parent_tokens,
                        "reusable_parent_prefix_tokens": run_reusable_tokens,
                        "reusable_group_prefix_tokens": run_reusable_group_tokens,
                        "reusable_prefix_fraction_of_candidate_tokens": (
                            run_reusable_tokens / run_candidate_tokens if run_candidate_tokens > 0 else 0.0
                        ),
                        "reusable_fraction_of_parent_prefix_tokens": (
                            run_reusable_tokens / run_parent_tokens if run_parent_tokens > 0 else 0.0
                        ),
                        "reusable_group_prefix_fraction_of_candidate_tokens": (
                            run_reusable_group_tokens / run_candidate_tokens if run_candidate_tokens > 0 else 0.0
                        ),
                        "weighted_pairwise_lcp_ratio": (
                            run_pairwise_lcp_weighted / run_pairs_used if run_pairs_used > 0 else 0.0
                        ),
                        "mean_duplicate_rate": safe_mean([float(row["duplicate_rate"]) for row in run_step_rows]),
                    }
                )

    if not step_rows:
        raise ValueError(
            "No step rows were analyzed. Check that logs contain non-naive ToT runs with steps/new_ys."
        )

    total_candidate_tokens = sum(float(row["total_candidate_tokens"]) for row in step_rows)
    total_parent_tokens = sum(float(row["total_parent_prefix_tokens"]) for row in step_rows)
    total_reusable_tokens = sum(float(row["reusable_parent_prefix_tokens"]) for row in step_rows)
    total_reusable_group_tokens = sum(float(row["reusable_group_prefix_tokens"]) for row in step_rows)

    total_pair_count_used = sum(int(row["pair_count_used"]) for row in step_rows)
    weighted_pairwise_ratio_sum = sum(
        float(row["pairwise_mean_lcp_ratio"]) * int(row["pair_count_used"]) for row in step_rows
    )

    total_sibling_pair_count_used = sum(int(row["sibling_pair_count_used"]) for row in step_rows)
    weighted_sibling_ratio_sum = sum(
        float(row["sibling_suffix_mean_lcp_ratio"]) * int(row["sibling_pair_count_used"]) for row in step_rows
    )

    top_steps = sorted(
        step_rows,
        key=lambda row: float(row["reusable_prefix_fraction_of_candidate_tokens"]),
        reverse=True,
    )[: args.top_k]

    summary: Dict[str, object] = {
        "config": {
            "tokenizer": args.tokenizer,
            "max_pairs": args.max_pairs,
            "min_candidates": args.min_candidates,
            "seed": args.seed,
            "inputs": args.inputs,
            "resolved_input_files": paths,
        },
        "totals": {
            "files": len(paths),
            "runs": len(run_rows),
            "steps": len(step_rows),
            "candidate_tokens": total_candidate_tokens,
            "parent_prefix_tokens": total_parent_tokens,
            "reusable_parent_prefix_tokens": total_reusable_tokens,
        },
        "aggregate": {
            "mean_num_candidates": safe_mean([float(row["num_candidates"]) for row in step_rows]),
            "mean_duplicate_rate": safe_mean([float(row["duplicate_rate"]) for row in step_rows]),
            "mean_parent_assignment_miss_rate": safe_mean(
                [float(row["parent_assignment_miss_rate"]) for row in step_rows]
            ),
            "mean_reusable_prefix_fraction_of_candidate_tokens": safe_mean(
                [float(row["reusable_prefix_fraction_of_candidate_tokens"]) for row in step_rows]
            ),
            "mean_reusable_group_prefix_fraction_of_candidate_tokens": safe_mean(
                [float(row["reusable_group_prefix_fraction_of_candidate_tokens"]) for row in step_rows]
            ),
            "corpus_reusable_prefix_fraction_of_candidate_tokens": (
                total_reusable_tokens / total_candidate_tokens if total_candidate_tokens > 0 else 0.0
            ),
            "corpus_reusable_group_prefix_fraction_of_candidate_tokens": (
                total_reusable_group_tokens / total_candidate_tokens if total_candidate_tokens > 0 else 0.0
            ),
            "corpus_reusable_fraction_of_parent_prefix_tokens": (
                total_reusable_tokens / total_parent_tokens if total_parent_tokens > 0 else 0.0
            ),
            "weighted_pairwise_lcp_ratio": (
                weighted_pairwise_ratio_sum / total_pair_count_used if total_pair_count_used > 0 else 0.0
            ),
            "weighted_sibling_suffix_lcp_ratio": (
                weighted_sibling_ratio_sum / total_sibling_pair_count_used
                if total_sibling_pair_count_used > 0
                else 0.0
            ),
        },
        "interpretation": {
            "reusable_prefix_fraction_of_candidate_tokens": (
                "Upper-bound fraction of candidate token compute that could be skipped "
                "if sibling candidates reuse cached parent prefixes."
            ),
            "reusable_group_prefix_fraction_of_candidate_tokens": (
                "Upper-bound fraction of candidate token compute that could be skipped "
                "if siblings also share any additional common prefix beyond the parent."
            ),
            "reusable_fraction_of_parent_prefix_tokens": (
                "Fraction of parent-prefix compute that is redundant across siblings. "
                "Higher means stronger motivation for prefix sharing."
            ),
            "pairwise_mean_lcp_ratio": (
                "Average normalized longest-common-prefix overlap among all candidates in a step."
            ),
            "sibling_suffix_mean_lcp_ratio": (
                "Additional overlap among siblings after removing inherited parent prefixes."
            ),
        },
        "top_steps_by_reusable_prefix_fraction": [
            {
                "source_file": row["source_file"],
                "record_idx": row["record_idx"],
                "step": row["step"],
                "num_candidates": row["num_candidates"],
                "reusable_prefix_fraction_of_candidate_tokens": row[
                    "reusable_prefix_fraction_of_candidate_tokens"
                ],
                "reusable_group_prefix_fraction_of_candidate_tokens": row[
                    "reusable_group_prefix_fraction_of_candidate_tokens"
                ],
                "reusable_fraction_of_parent_prefix_tokens": row[
                    "reusable_fraction_of_parent_prefix_tokens"
                ],
                "pairwise_mean_lcp_ratio": row["pairwise_mean_lcp_ratio"],
                "sibling_suffix_mean_lcp_ratio": row["sibling_suffix_mean_lcp_ratio"],
            }
            for row in top_steps
        ],
    }

    return step_rows, run_rows, summary


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze overlap structure in Tree-of-Thought logs and estimate "
            "prefix-sharing opportunity."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Log files or glob patterns (example: logs/text/*.json or logs/**/*.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/thought_overlap",
        help="Directory where step_metrics.csv, run_metrics.csv, and summary.json are written",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["word", "char"],
        default="word",
        help="Tokenization granularity for overlap computations",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=50000,
        help="Max pairwise comparisons per step/group; sampled if exceeded",
    )
    parser.add_argument(
        "--min_candidates",
        type=int,
        default=2,
        help="Ignore steps with fewer than this many candidates",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top-overlap steps to include in summary.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for pair subsampling reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_rows, run_rows, summary = run_analysis(args)

    os.makedirs(args.output_dir, exist_ok=True)
    step_path = os.path.join(args.output_dir, "step_metrics.csv")
    run_path = os.path.join(args.output_dir, "run_metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    write_csv(step_path, step_rows)
    write_csv(run_path, run_rows)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Analyzed {summary['totals']['steps']} steps across {summary['totals']['runs']} runs.")
    print(f"Wrote: {step_path}")
    print(f"Wrote: {run_path}")
    print(f"Wrote: {summary_path}")
    print(
        "Corpus reusable_prefix_fraction_of_candidate_tokens = "
        f"{summary['aggregate']['corpus_reusable_prefix_fraction_of_candidate_tokens']:.4f}"
    )
    print(
        "Corpus reusable_group_prefix_fraction_of_candidate_tokens = "
        f"{summary['aggregate']['corpus_reusable_group_prefix_fraction_of_candidate_tokens']:.4f}"
    )


if __name__ == "__main__":
    main()
