import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from ogb.linkproppred import Evaluator
from tqdm import tqdm

from src.biokg_data import load_biokg, split_to_global


Template = Tuple[int, int]


def metric_mean(x) -> float:
    return float(np.asarray(x).mean())


def build_out_neighbors(head: np.ndarray, rel: np.ndarray, tail: np.ndarray):
    out_by_rel = defaultdict(lambda: defaultdict(list))
    for h, r, t in zip(head.tolist(), rel.tolist(), tail.tolist()):
        out_by_rel[int(h)][int(r)].append(int(t))
    return out_by_rel


def reverse_out_neighbors(out_by_rel):
    rev = defaultdict(lambda: defaultdict(list))
    for h, rel_dict in out_by_rel.items():
        for r, tails in rel_dict.items():
            for t in tails:
                rev[t][r].append(h)
    return rev


def build_sparse_adjacency_by_relation(
    head: np.ndarray,
    rel: np.ndarray,
    tail: np.ndarray,
    num_entities: int,
    device: torch.device,
):
    edges_by_rel = defaultdict(lambda: [[], []])
    for h, r, t in zip(head.tolist(), rel.tolist(), tail.tolist()):
        bucket = edges_by_rel[int(r)]
        bucket[0].append(int(h))
        bucket[1].append(int(t))

    adj_by_rel = {}
    for r, (hs, ts) in edges_by_rel.items():
        indices = torch.tensor([hs, ts], dtype=torch.long, device=device)
        values = torch.ones(len(hs), dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_entities, num_entities),
            device=device,
        ).coalesce()
        adj_by_rel[r] = adj
    return adj_by_rel


def mine_length2_templates(
    head: np.ndarray,
    rel: np.ndarray,
    tail: np.ndarray,
    out_by_rel,
    max_examples_per_relation: int,
    max_templates_per_relation: int,
    min_count: int,
    seed: int,
    show_progress: bool,
) -> Dict[int, List[Tuple[Template, int]]]:
    rng = np.random.default_rng(seed)
    indices_by_rel = defaultdict(list)
    for i, r in enumerate(rel.tolist()):
        indices_by_rel[int(r)].append(i)

    templates_by_target_rel: Dict[int, List[Tuple[Template, int]]] = {}

    relation_items = list(indices_by_rel.items())
    if show_progress:
        relation_items = tqdm(relation_items, desc="Mining templates", unit="rel")

    for target_rel, idxs in relation_items:
        idxs = np.asarray(idxs, dtype=np.int64)
        if len(idxs) > max_examples_per_relation:
            idxs = rng.choice(idxs, size=max_examples_per_relation, replace=False)

        counts: Counter = Counter()
        for idx in idxs.tolist():
            h = int(head[idx])
            t = int(tail[idx])
            rel_neighbors = out_by_rel.get(h, {})
            for r1, mid_nodes in rel_neighbors.items():
                for mid in mid_nodes:
                    mid_neighbors = out_by_rel.get(mid, {})
                    for r2, dst_nodes in mid_neighbors.items():
                        if t in dst_nodes:
                            counts[(r1, r2)] += 1

        filtered = [(tpl, c) for tpl, c in counts.items() if c >= min_count]
        filtered.sort(key=lambda x: x[1], reverse=True)
        templates_by_target_rel[target_rel] = filtered[:max_templates_per_relation]

    return templates_by_target_rel


def score_query_length2(
    h: int,
    templates: List[Tuple[Template, int]],
    out_by_rel,
) -> Dict[int, float]:
    scores = defaultdict(float)
    rel_neighbors = out_by_rel.get(h, {})
    for (r1, r2), weight in templates:
        mids = rel_neighbors.get(r1, [])
        if not mids:
            continue
        for mid in mids:
            tails = out_by_rel.get(mid, {}).get(r2, [])
            for t in tails:
                scores[t] += float(weight)
    return scores


def score_query_length2_gpu(
    h: int,
    templates: List[Tuple[Template, int]],
    adj_by_rel,
    num_entities: int,
    device: torch.device,
) -> torch.Tensor:
    src = torch.zeros((1, num_entities), dtype=torch.float32, device=device)
    src[0, h] = 1.0
    scores = torch.zeros((1, num_entities), dtype=torch.float32, device=device)

    for (r1, r2), weight in templates:
        a1 = adj_by_rel.get(r1)
        a2 = adj_by_rel.get(r2)
        if a1 is None or a2 is None:
            continue
        mid = torch.sparse.mm(src, a1)
        dst = torch.sparse.mm(mid, a2)
        scores = scores + float(weight) * dst

    return scores.squeeze(0)


def evaluate_split(split_global, templates_by_rel, out_by_rel, reverse_out, evaluator: Evaluator, show_progress: bool, label: str):
    heads = split_global["head"]
    tails = split_global["tail"]
    rels = split_global["relation"]
    tail_neg = split_global["tail_neg"]
    head_neg = split_global["head_neg"]

    pos_tail_scores = np.zeros(len(heads), dtype=np.float32)
    neg_tail_scores = np.zeros_like(tail_neg, dtype=np.float32)
    pos_head_scores = np.zeros(len(heads), dtype=np.float32)
    neg_head_scores = np.zeros_like(head_neg, dtype=np.float32)

    loop = range(len(heads))
    if show_progress:
        loop = tqdm(loop, desc=f"Eval {label}", unit="triple")

    for i in loop:
        h = int(heads[i])
        t = int(tails[i])
        r = int(rels[i])

        templates = templates_by_rel.get(r, [])
        score_map_tail = score_query_length2(h, templates, out_by_rel)
        pos_tail_scores[i] = score_map_tail.get(t, 0.0)
        neg_tail_scores[i] = np.asarray([score_map_tail.get(int(x), 0.0) for x in tail_neg[i]], dtype=np.float32)

        score_map_head = score_query_length2(t, templates, reverse_out)
        pos_head_scores[i] = score_map_head.get(h, 0.0)
        neg_head_scores[i] = np.asarray([score_map_head.get(int(x), 0.0) for x in head_neg[i]], dtype=np.float32)

    tail_res = evaluator.eval(
        {
            "y_pred_pos": torch.from_numpy(pos_tail_scores),
            "y_pred_neg": torch.from_numpy(neg_tail_scores),
        }
    )
    head_res = evaluator.eval(
        {
            "y_pred_pos": torch.from_numpy(pos_head_scores),
            "y_pred_neg": torch.from_numpy(neg_head_scores),
        }
    )

    return {
        "mrr": float((metric_mean(tail_res["mrr_list"]) + metric_mean(head_res["mrr_list"])) / 2.0),
        "hits@1": float((metric_mean(tail_res["hits@1_list"]) + metric_mean(head_res["hits@1_list"])) / 2.0),
        "hits@3": float((metric_mean(tail_res["hits@3_list"]) + metric_mean(head_res["hits@3_list"])) / 2.0),
        "hits@10": float((metric_mean(tail_res["hits@10_list"]) + metric_mean(head_res["hits@10_list"])) / 2.0),
    }


def evaluate_split_gpu(
    split_global,
    templates_by_rel,
    adj_by_rel,
    reverse_adj_by_rel,
    num_entities: int,
    evaluator: Evaluator,
    device: torch.device,
    show_progress: bool,
    label: str,
):
    heads = split_global["head"]
    tails = split_global["tail"]
    rels = split_global["relation"]
    tail_neg = split_global["tail_neg"]
    head_neg = split_global["head_neg"]

    pos_tail_scores = np.zeros(len(heads), dtype=np.float32)
    neg_tail_scores = np.zeros_like(tail_neg, dtype=np.float32)
    pos_head_scores = np.zeros(len(heads), dtype=np.float32)
    neg_head_scores = np.zeros_like(head_neg, dtype=np.float32)

    tail_query_groups = defaultdict(list)
    for i, (h, r) in enumerate(zip(heads.tolist(), rels.tolist())):
        tail_query_groups[(int(h), int(r))].append(i)

    tail_items = list(tail_query_groups.items())
    if show_progress:
        tail_items = tqdm(tail_items, desc=f"Eval {label} tail", unit="query")

    for (h, r), idxs in tail_items:
        templates = templates_by_rel.get(r, [])
        if not templates:
            continue
        score_vec = score_query_length2_gpu(h, templates, adj_by_rel, num_entities, device)
        idx_arr = np.array(idxs, dtype=np.int64)

        pos_ids = torch.from_numpy(tails[idx_arr]).to(device)
        pos_tail_scores[idx_arr] = score_vec[pos_ids].detach().cpu().numpy()

        neg_ids = torch.from_numpy(tail_neg[idx_arr]).to(device)
        neg_tail_scores[idx_arr] = score_vec[neg_ids].detach().cpu().numpy()

    head_query_groups = defaultdict(list)
    for i, (t, r) in enumerate(zip(tails.tolist(), rels.tolist())):
        head_query_groups[(int(t), int(r))].append(i)

    head_items = list(head_query_groups.items())
    if show_progress:
        head_items = tqdm(head_items, desc=f"Eval {label} head", unit="query")

    for (t, r), idxs in head_items:
        templates = templates_by_rel.get(r, [])
        if not templates:
            continue
        score_vec = score_query_length2_gpu(t, templates, reverse_adj_by_rel, num_entities, device)
        idx_arr = np.array(idxs, dtype=np.int64)

        pos_ids = torch.from_numpy(heads[idx_arr]).to(device)
        pos_head_scores[idx_arr] = score_vec[pos_ids].detach().cpu().numpy()

        neg_ids = torch.from_numpy(head_neg[idx_arr]).to(device)
        neg_head_scores[idx_arr] = score_vec[neg_ids].detach().cpu().numpy()

    tail_res = evaluator.eval(
        {
            "y_pred_pos": torch.from_numpy(pos_tail_scores),
            "y_pred_neg": torch.from_numpy(neg_tail_scores),
        }
    )
    head_res = evaluator.eval(
        {
            "y_pred_pos": torch.from_numpy(pos_head_scores),
            "y_pred_neg": torch.from_numpy(neg_head_scores),
        }
    )

    return {
        "mrr": float((metric_mean(tail_res["mrr_list"]) + metric_mean(head_res["mrr_list"])) / 2.0),
        "hits@1": float((metric_mean(tail_res["hits@1_list"]) + metric_mean(head_res["hits@1_list"])) / 2.0),
        "hits@3": float((metric_mean(tail_res["hits@3_list"]) + metric_mean(head_res["hits@3_list"])) / 2.0),
        "hits@10": float((metric_mean(tail_res["hits@10_list"]) + metric_mean(head_res["hits@10_list"])) / 2.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1 - Subset 1: Type-aware metapath heuristic baseline")
    parser.add_argument("--max_examples_per_relation", type=int, default=1500)
    parser.add_argument("--max_templates_per_relation", type=int, default=24)
    parser.add_argument("--min_template_count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/stage1_subset1_metrics.json")
    args = parser.parse_args()
    show_progress = not args.no_progress

    start = time.time()
    print("[stage1-subset1] loading dataset...", flush=True)

    ctx = load_biokg()
    print("[stage1-subset1] converting train split ids...", flush=True)
    split_train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    print("[stage1-subset1] converting valid split ids...", flush=True)
    split_valid = split_to_global(ctx.split_edge["valid"], ctx.type_offsets)
    print("[stage1-subset1] converting test split ids...", flush=True)
    split_test = split_to_global(ctx.split_edge["test"], ctx.type_offsets)

    out_by_rel = build_out_neighbors(split_train["head"], split_train["relation"], split_train["tail"])
    reverse_out = reverse_out_neighbors(out_by_rel)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[stage1-subset1] using device={device}", flush=True)
    print("[stage1-subset1] mining templates...", flush=True)
    templates_by_rel = mine_length2_templates(
        split_train["head"],
        split_train["relation"],
        split_train["tail"],
        out_by_rel,
        max_examples_per_relation=args.max_examples_per_relation,
        max_templates_per_relation=args.max_templates_per_relation,
        min_count=args.min_template_count,
        seed=args.seed,
        show_progress=show_progress,
    )

    evaluator = Evaluator(name="ogbl-biokg")
    adj_by_rel: Optional[Dict[int, torch.Tensor]] = None
    reverse_adj_by_rel: Optional[Dict[int, torch.Tensor]] = None
    if device.type == "cuda":
        print("[stage1-subset1] building sparse adjacency on GPU...", flush=True)
        adj_by_rel = build_sparse_adjacency_by_relation(
            split_train["head"],
            split_train["relation"],
            split_train["tail"],
            ctx.num_entities,
            device,
        )
        reverse_adj_by_rel = build_sparse_adjacency_by_relation(
            split_train["tail"],
            split_train["relation"],
            split_train["head"],
            ctx.num_entities,
            device,
        )
        adj_cuda = cast(Dict[int, torch.Tensor], adj_by_rel)
        rev_adj_cuda = cast(Dict[int, torch.Tensor], reverse_adj_by_rel)
        valid_metrics = evaluate_split_gpu(
            split_valid,
            templates_by_rel,
            adj_cuda,
            rev_adj_cuda,
            ctx.num_entities,
            evaluator,
            device,
            show_progress=show_progress,
            label="valid",
        )
        test_metrics = evaluate_split_gpu(
            split_test,
            templates_by_rel,
            adj_cuda,
            rev_adj_cuda,
            ctx.num_entities,
            evaluator,
            device,
            show_progress=show_progress,
            label="test",
        )
    else:
        valid_metrics = evaluate_split(
            split_valid,
            templates_by_rel,
            out_by_rel,
            reverse_out,
            evaluator,
            show_progress=show_progress,
            label="valid",
        )
        test_metrics = evaluate_split(
            split_test,
            templates_by_rel,
            out_by_rel,
            reverse_out,
            evaluator,
            show_progress=show_progress,
            label="test",
        )

    per_relation_mrr = {}
    relation_ids = np.unique(split_test["relation"])
    relation_iter = relation_ids
    if show_progress:
        relation_iter = tqdm(relation_ids, desc="Per-relation MRR", unit="rel")

    for r in relation_iter:
        idx = np.where(split_test["relation"] == r)[0]
        sliced = {
            "head": split_test["head"][idx],
            "tail": split_test["tail"][idx],
            "relation": split_test["relation"][idx],
            "head_neg": split_test["head_neg"][idx],
            "tail_neg": split_test["tail_neg"][idx],
        }
        if device.type == "cuda":
            assert adj_by_rel is not None and reverse_adj_by_rel is not None
            adj_cuda = cast(Dict[int, torch.Tensor], adj_by_rel)
            rev_adj_cuda = cast(Dict[int, torch.Tensor], reverse_adj_by_rel)
            per_relation_mrr[int(r)] = evaluate_split_gpu(
                sliced,
                templates_by_rel,
                adj_cuda,
                rev_adj_cuda,
                ctx.num_entities,
                evaluator,
                device,
                show_progress=False,
                label=f"rel-{int(r)}",
            )["mrr"]
        else:
            per_relation_mrr[int(r)] = evaluate_split(
                sliced,
                templates_by_rel,
                out_by_rel,
                reverse_out,
                evaluator,
                show_progress=False,
                label=f"rel-{int(r)}",
            )["mrr"]

    payload = {
        "model": "metapath_length2_type_aware",
        "config": {
            "max_examples_per_relation": args.max_examples_per_relation,
            "max_templates_per_relation": args.max_templates_per_relation,
            "min_template_count": args.min_template_count,
            "seed": args.seed,
            "device": str(device),
        },
        "valid": valid_metrics,
        "test": test_metrics,
        "test_mrr_by_relation": per_relation_mrr,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.time() - start
    print(f"[stage1-subset1] done in {elapsed / 60.0:.2f} minutes", flush=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
