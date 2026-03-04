import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.biokg_data import load_biokg, split_to_global
from stage1_subset1_heuristic import build_out_neighbors, reverse_out_neighbors, score_query_length2


def load_metrics_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def relation_freq_slices(train_relation: np.ndarray) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
    rel_ids, counts = np.unique(train_relation, return_counts=True)
    rel_count = {int(r): int(c) for r, c in zip(rel_ids.tolist(), counts.tolist())}

    cvals = np.array(list(rel_count.values()), dtype=float)
    q1, q2 = np.percentile(cvals, [33.3, 66.6])

    buckets = {"rare": [], "medium": [], "frequent": []}
    for r, c in rel_count.items():
        if c <= q1:
            buckets["rare"].append(r)
        elif c <= q2:
            buckets["medium"].append(r)
        else:
            buckets["frequent"].append(r)
    return rel_count, buckets


def extract_relation_mrr(metrics_payload: dict) -> Dict[int, float]:
    if "test_mrr_by_relation_mean" in metrics_payload:
        src = metrics_payload["test_mrr_by_relation_mean"]
    elif "test_mrr_by_relation" in metrics_payload:
        src = metrics_payload["test_mrr_by_relation"]
    else:
        return {}
    return {int(k): float(v) for k, v in src.items()}


def summarize_frequency_slices(model_name: str, rel_mrr: Dict[int, float], rel_buckets: Dict[str, List[int]]):
    out = {"model": model_name, "slice_mrr": {}}
    for bucket, rels in rel_buckets.items():
        vals = [rel_mrr[r] for r in rels if r in rel_mrr]
        out["slice_mrr"][bucket] = float(np.mean(vals)) if vals else None
    return out


def mine_templates_fast(
    head: np.ndarray,
    rel: np.ndarray,
    tail: np.ndarray,
    out_by_rel,
    max_examples_per_relation: int,
    max_templates_per_relation: int,
    min_count: int,
    seed: int,
    target_rel_ids: List[int],
) -> Dict[int, List[Tuple[Tuple[int, int], int]]]:
    rng = np.random.default_rng(seed)

    out_sets = {}
    for src, rel_dict in out_by_rel.items():
        out_sets[src] = {r: set(dsts) for r, dsts in rel_dict.items()}

    indices_by_rel: Dict[int, List[int]] = {}
    for i, r in enumerate(rel.tolist()):
        ri = int(r)
        if ri in target_rel_ids:
            indices_by_rel.setdefault(ri, []).append(i)

    templates = {}
    for target_rel in target_rel_ids:
        idxs = np.asarray(indices_by_rel.get(target_rel, []), dtype=np.int64)
        if idxs.size == 0:
            templates[target_rel] = []
            continue
        if idxs.size > max_examples_per_relation:
            idxs = rng.choice(idxs, size=max_examples_per_relation, replace=False)

        counts: Dict[Tuple[int, int], int] = {}
        for idx in idxs.tolist():
            h = int(head[idx])
            t = int(tail[idx])
            for r1, mids in out_by_rel.get(h, {}).items():
                for mid in mids:
                    for r2, dst_set in out_sets.get(mid, {}).items():
                        if t in dst_set:
                            key = (int(r1), int(r2))
                            counts[key] = counts.get(key, 0) + 1

        filtered = [(k, c) for k, c in counts.items() if c >= min_count]
        filtered.sort(key=lambda x: x[1], reverse=True)
        templates[target_rel] = filtered[:max_templates_per_relation]
    return templates


def compute_entity_degree(train_head: np.ndarray, train_tail: np.ndarray, num_entities: int) -> np.ndarray:
    deg = np.zeros(num_entities, dtype=np.int64)
    np.add.at(deg, train_head, 1)
    np.add.at(deg, train_tail, 1)
    return deg


def assign_degree_bucket(value: int, q_low: float, q_high: float) -> str:
    if value <= q_low:
        return "low"
    if value <= q_high:
        return "medium"
    return "high"


def analyze_heuristic_degree_and_errors(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    num_entities: int,
    rel_count: Dict[int, int],
    max_test_samples: int,
    template_examples: int,
    template_topk: int,
    template_min_count: int,
    template_train_edges: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    train_idx = np.arange(train["head"].shape[0])
    if template_train_edges > 0 and template_train_edges < train_idx.size:
        train_idx = rng.choice(train_idx, size=template_train_edges, replace=False)

    train_head_m = train["head"][train_idx]
    train_rel_m = train["relation"][train_idx]
    train_tail_m = train["tail"][train_idx]

    n_test = test["head"].shape[0]
    if max_test_samples > 0 and max_test_samples < n_test:
        idx = rng.choice(n_test, size=max_test_samples, replace=False)
    else:
        idx = np.arange(n_test)

    out_by_rel = build_out_neighbors(train_head_m, train_rel_m, train_tail_m)
    reverse_out = reverse_out_neighbors(out_by_rel)

    target_rel_ids = sorted(np.unique(test["relation"][idx]).astype(int).tolist())
    templates_by_rel = mine_templates_fast(
        train_head_m,
        train_rel_m,
        train_tail_m,
        out_by_rel,
        max_examples_per_relation=template_examples,
        max_templates_per_relation=template_topk,
        min_count=template_min_count,
        seed=seed,
        target_rel_ids=target_rel_ids,
    )

    deg = compute_entity_degree(train["head"], train["tail"], num_entities)
    non_zero_deg = deg[deg > 0]
    q_low, q_high = np.percentile(non_zero_deg, [33.3, 66.6])
    q_hub = np.percentile(non_zero_deg, 90.0)

    degree_slice_rr = {"low": [], "medium": [], "high": []}
    high_conf_fp = []
    high_conf_fn = []

    rel_count_values = np.array(list(rel_count.values()), dtype=float)
    rel_rare_thr = float(np.percentile(rel_count_values, 25.0))

    for i in idx.tolist():
        h = int(test["head"][i])
        t = int(test["tail"][i])
        r = int(test["relation"][i])
        templates = templates_by_rel.get(r, [])

        tail_scores = score_query_length2(h, templates, out_by_rel)
        pos_t = float(tail_scores.get(t, 0.0))
        neg_tail_ids = test["tail_neg"][i]
        neg_tail_scores = np.asarray([tail_scores.get(int(x), 0.0) for x in neg_tail_ids], dtype=np.float32)

        rank_tail = 1.0 + float(np.sum(neg_tail_scores > pos_t)) + 0.5 * float(np.sum(neg_tail_scores == pos_t))
        rr_tail = 1.0 / rank_tail
        b_tail = assign_degree_bucket(int(deg[t]), q_low, q_high)
        degree_slice_rr[b_tail].append(rr_tail)

        if neg_tail_scores.size > 0:
            top_idx = int(np.argmax(neg_tail_scores))
            top_neg_score = float(neg_tail_scores[top_idx])
            top_neg_id = int(neg_tail_ids[top_idx])
            if top_neg_score > pos_t:
                true_deg = int(deg[t])
                pred_deg = int(deg[top_neg_id])
                if pred_deg >= q_hub and pred_deg > max(2 * true_deg, 1):
                    label = "hub_bias"
                elif rel_count.get(r, 0) <= rel_rare_thr:
                    label = "rare_relation"
                elif pos_t == 0.0:
                    label = "no_supporting_path"
                else:
                    label = "ambiguous_signal"
                high_conf_fp.append(
                    {
                        "query_index": i,
                        "type": "tail_query",
                        "head": h,
                        "relation": r,
                        "true_tail": t,
                        "pred_tail": top_neg_id,
                        "true_score": pos_t,
                        "pred_score": top_neg_score,
                        "margin": top_neg_score - pos_t,
                        "classification": label,
                    }
                )

        head_scores = score_query_length2(t, templates, reverse_out)
        pos_h = float(head_scores.get(h, 0.0))
        neg_head_ids = test["head_neg"][i]
        neg_head_scores = np.asarray([head_scores.get(int(x), 0.0) for x in neg_head_ids], dtype=np.float32)

        rank_head = 1.0 + float(np.sum(neg_head_scores > pos_h)) + 0.5 * float(np.sum(neg_head_scores == pos_h))
        rr_head = 1.0 / rank_head
        b_head = assign_degree_bucket(int(deg[h]), q_low, q_high)
        degree_slice_rr[b_head].append(rr_head)

        if rr_tail < 0.1:
            if rel_count.get(r, 0) <= rel_rare_thr:
                label = "rare_relation"
            elif deg[t] <= q_low:
                label = "cold_start_entity"
            elif pos_t == 0.0:
                label = "no_supporting_path"
            else:
                label = "ambiguous_signal"
            high_conf_fn.append(
                {
                    "query_index": i,
                    "type": "tail_query",
                    "head": h,
                    "relation": r,
                    "true_tail": t,
                    "true_score": pos_t,
                    "rr": rr_tail,
                    "classification": label,
                }
            )

        if rr_head < 0.1:
            if rel_count.get(r, 0) <= rel_rare_thr:
                label = "rare_relation"
            elif deg[h] <= q_low:
                label = "cold_start_entity"
            elif pos_h == 0.0:
                label = "no_supporting_path"
            else:
                label = "ambiguous_signal"
            high_conf_fn.append(
                {
                    "query_index": i,
                    "type": "head_query",
                    "tail": t,
                    "relation": r,
                    "true_head": h,
                    "true_score": pos_h,
                    "rr": rr_head,
                    "classification": label,
                }
            )

    high_conf_fp = sorted(high_conf_fp, key=lambda x: x["margin"], reverse=True)
    high_conf_fn = sorted(high_conf_fn, key=lambda x: x["rr"])  # lowest RR first

    degree_mrr = {
        k: float(np.mean(v)) if len(v) > 0 else None for k, v in degree_slice_rr.items()
    }

    return {
        "degree_slice_mrr": degree_mrr,
        "high_confidence_false_positives": high_conf_fp[:25],
        "high_confidence_false_negatives": high_conf_fn[:25],
        "num_test_queries_used": int(len(idx)),
    }


def to_markdown(payload: dict) -> str:
    lines = [
        "# Stage 1 - Subset 4 Diagnostics",
        "",
        "## Relation Frequency Slices",
        "",
        "| Model | Rare | Medium | Frequent |",
        "|---|---:|---:|---:|",
    ]

    for row in payload["relation_frequency_slices"]:
        sm = row["slice_mrr"]
        def fmt(v):
            return "n/a" if v is None else f"{v:.4f}"
        lines.append(f"| {row['model']} | {fmt(sm.get('rare'))} | {fmt(sm.get('medium'))} | {fmt(sm.get('frequent'))} |")

    lines.extend(
        [
            "",
            "## Degree Slices (heuristic diagnostic run)",
            "",
            f"- Low degree MRR: {payload['degree_and_error_analysis']['degree_slice_mrr'].get('low')}",
            f"- Medium degree MRR: {payload['degree_and_error_analysis']['degree_slice_mrr'].get('medium')}",
            f"- High degree MRR: {payload['degree_and_error_analysis']['degree_slice_mrr'].get('high')}",
            "",
            "## Error Case Snapshots",
            "",
            f"- Collected high-confidence false positives: {len(payload['degree_and_error_analysis']['high_confidence_false_positives'])}",
            f"- Collected high-confidence false negatives: {len(payload['degree_and_error_analysis']['high_confidence_false_negatives'])}",
            "",
            "The full structured error records are available in the JSON output.",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 - Subset 4 diagnostics")
    parser.add_argument("--subset1_metrics", type=str, default="outputs/stage1_subset1_metrics.json")
    parser.add_argument("--distmult_metrics", type=str, default="outputs/stage1_subset2_distmult_metrics.json")
    parser.add_argument("--complex_metrics", type=str, default="outputs/stage1_subset2_complex_metrics.json")
    parser.add_argument("--subset3_metrics", type=str, default="outputs/stage1_subset3_rgcn_metrics.json")

    parser.add_argument("--max_test_samples", type=int, default=3000)
    parser.add_argument("--template_examples", type=int, default=500)
    parser.add_argument("--template_topk", type=int, default=16)
    parser.add_argument("--template_min_count", type=int, default=4)
    parser.add_argument("--template_train_edges", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_json", type=str, default="outputs/stage1_subset4_diagnostics.json")
    parser.add_argument("--output_md", type=str, default="outputs/stage1_subset4_diagnostics.md")
    args = parser.parse_args()

    ctx = load_biokg()
    train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    test = split_to_global(ctx.split_edge["test"], ctx.type_offsets)

    rel_count, rel_buckets = relation_freq_slices(train["relation"])

    model_payloads = []
    for path, name in [
        (args.subset1_metrics, "subset1_heuristic"),
        (args.distmult_metrics, "subset2_distmult"),
        (args.complex_metrics, "subset2_complex"),
        (args.subset3_metrics, "subset3_rgcn"),
    ]:
        payload = load_metrics_if_exists(path)
        if payload is None:
            continue
        rel_mrr = extract_relation_mrr(payload)
        if len(rel_mrr) == 0 and "subset3" in payload and "base" in payload["subset3"]:
            rel_mrr = extract_relation_mrr(payload["subset3"]["base"])
        if len(rel_mrr) == 0:
            continue
        model_payloads.append(summarize_frequency_slices(name, rel_mrr, rel_buckets))

    degree_and_error = analyze_heuristic_degree_and_errors(
        train=train,
        test=test,
        num_entities=ctx.num_entities,
        rel_count=rel_count,
        max_test_samples=args.max_test_samples,
        template_examples=args.template_examples,
        template_topk=args.template_topk,
        template_min_count=args.template_min_count,
        template_train_edges=args.template_train_edges,
        seed=args.seed,
    )

    output = {
        "relation_frequency_slices": model_payloads,
        "relation_frequency_bucket_sizes": {k: len(v) for k, v in rel_buckets.items()},
        "degree_and_error_analysis": degree_and_error,
        "config": {
            "max_test_samples": args.max_test_samples,
            "template_examples": args.template_examples,
            "template_topk": args.template_topk,
            "template_min_count": args.template_min_count,
            "template_train_edges": args.template_train_edges,
            "seed": args.seed,
        },
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    md = to_markdown(output)
    out_dir_md = os.path.dirname(args.output_md)
    if out_dir_md:
        os.makedirs(out_dir_md, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "relation_models": [x["model"] for x in model_payloads],
        "degree_slice_mrr": degree_and_error["degree_slice_mrr"],
    }, indent=2))


if __name__ == "__main__":
    main()
