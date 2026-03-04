import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from stage1_subset2_kge import ComplEx, DistMult
from src.biokg_data import (
    load_biokg,
    split_to_global,
    type_id_buckets,
)


def sample_type_aware_negatives(
    type_names: np.ndarray,
    type_offsets: Dict[str, int],
    num_nodes_by_type: Dict[str, int],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros((len(type_names), k), dtype=np.int64)
    for i, t in enumerate(type_names):
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        t = str(t)
        size = num_nodes_by_type[t]
        offset = type_offsets[t]
        out[i] = rng.integers(0, size, size=k, endpoint=False) + offset
    return out


def build_model(name: str, num_entities: int, num_relations: int, dim: int):
    if name.lower() == "distmult":
        return DistMult(num_entities, num_relations, dim)
    if name.lower() == "complex":
        return ComplEx(num_entities, num_relations, dim)
    raise ValueError(f"Unknown model: {name}")


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_select_best_model(dist_path: str, complex_path: str) -> str:
    d = read_json(dist_path)
    c = read_json(complex_path)
    if d is None and c is None:
        return "complex"
    if d is None:
        return "complex"
    if c is None:
        return "distmult"
    dm = float(d.get("test_mean", {}).get("mrr", -1.0))
    cm = float(c.get("test_mean", {}).get("mrr", -1.0))
    return "complex" if cm >= dm else "distmult"


def train_kge(
    model: torch.nn.Module,
    train: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    ctx,
    device: torch.device,
    epochs: int,
    batch_size: int,
    neg_k: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
    seed: int,
) -> Tuple[dict, Dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid = -1.0
    best_state = None
    history = []

    n = train["head"].shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        losses = []
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            h = torch.from_numpy(train["head"][idx]).to(device)
            t = torch.from_numpy(train["tail"][idx]).to(device)
            r = torch.from_numpy(train["relation"][idx]).to(device)

            tail_neg_np = sample_type_aware_negatives(
                train["tail_type"][idx],
                ctx.type_offsets,
                ctx.num_nodes_by_type,
                neg_k,
                rng,
            )
            head_neg_np = sample_type_aware_negatives(
                train["head_type"][idx],
                ctx.type_offsets,
                ctx.num_nodes_by_type,
                neg_k,
                rng,
            )
            tail_neg = torch.from_numpy(tail_neg_np).to(device)
            head_neg = torch.from_numpy(head_neg_np).to(device)

            pos = model.score(h, r, t)
            neg_t = model.score_tails(h, r, tail_neg)
            neg_h = model.score_heads(head_neg, r, t)

            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + torch.nn.functional.binary_cross_entropy_with_logits(neg_t, torch.zeros_like(neg_t))
                + torch.nn.functional.binary_cross_entropy_with_logits(neg_h, torch.zeros_like(neg_h))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        avg_loss = float(np.mean(losses))

        if epoch % eval_every == 0 or epoch == epochs:
            valid_mrr = quick_valid_mrr(model, valid, device, batch_size=512)
            history.append({"epoch": int(epoch), "train_loss": avg_loss, "valid_mrr": valid_mrr})
            if valid_mrr > best_valid:
                best_valid = valid_mrr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={epoch} loss={avg_loss:.4f} valid_mrr_proxy={valid_mrr:.4f} best={best_valid:.4f}")

    if best_state is None:
        raise RuntimeError("No valid checkpoint selected during training")

    return {"best_valid_mrr_proxy": best_valid, "history": history}, best_state


@torch.no_grad()
def quick_valid_mrr(model, split: Dict[str, np.ndarray], device: torch.device, batch_size: int = 512) -> float:
    model.eval()
    n = split["head"].shape[0]
    idx = np.arange(min(n, 4000), dtype=np.int64)

    rr = []
    for s in range(0, idx.size, batch_size):
        b = idx[s : s + batch_size]
        h = torch.from_numpy(split["head"][b]).to(device)
        t = torch.from_numpy(split["tail"][b]).to(device)
        r = torch.from_numpy(split["relation"][b]).to(device)
        tail_neg = torch.from_numpy(split["tail_neg"][b]).to(device)

        pos = model.score(h, r, t)
        neg = model.score_tails(h, r, tail_neg)
        greater = (neg > pos.unsqueeze(1)).sum(dim=1).float()
        equal = (neg == pos.unsqueeze(1)).sum(dim=1).float()
        rank = 1.0 + greater + 0.5 * equal
        rr.extend((1.0 / rank).detach().cpu().numpy().tolist())
    return float(np.mean(rr))


def relation_freq_buckets(train_relation: np.ndarray) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
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


def degree_buckets(train_head: np.ndarray, train_tail: np.ndarray, num_entities: int):
    deg = np.zeros(num_entities, dtype=np.int64)
    np.add.at(deg, train_head, 1)
    np.add.at(deg, train_tail, 1)
    non_zero = deg[deg > 0]
    q1, q2 = np.percentile(non_zero, [33.3, 66.6])
    return deg, float(q1), float(q2)


def deg_bucket(value: int, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


@torch.no_grad()
def generate_candidates(
    model,
    split: Dict[str, np.ndarray],
    train: Dict[str, np.ndarray],
    type_buckets: Dict[str, np.ndarray],
    k_values: List[int],
    device: torch.device,
    score_batch_size: int,
    out_path: str,
    max_queries: int,
) -> dict:
    model.eval()
    max_k = max(k_values)

    n = split["head"].shape[0]
    query_indices = np.arange(n, dtype=np.int64)
    if max_queries > 0 and max_queries < n:
        query_indices = query_indices[:max_queries]

    rel_count, rel_freq = relation_freq_buckets(train["relation"])
    deg, q1, q2 = degree_buckets(train["head"], train["tail"], int(deg_size_from_split(train, split)))

    coverage_overall = {k: 0 for k in k_values}
    by_rel_freq = {b: {k: 0 for k in k_values} for b in ["rare", "medium", "frequent"]}
    by_deg = {b: {k: 0 for k in k_values} for b in ["low", "medium", "high"]}
    denom_rel = {b: 0 for b in ["rare", "medium", "frequent"]}
    denom_deg = {b: 0 for b in ["low", "medium", "high"]}

    rel_to_bucket = {}
    for b, rels in rel_freq.items():
        for r in rels:
            rel_to_bucket[r] = b

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(query_indices.tolist(), desc="Generating top-K candidates", unit="query"):
            h = int(split["head"][i])
            r = int(split["relation"][i])
            true_t = int(split["tail"][i])
            t_type = split["tail_type"][i]
            if isinstance(t_type, bytes):
                t_type = t_type.decode("utf-8")
            t_type = str(t_type)
            candidates = type_buckets[t_type]

            h_t = torch.tensor([h], dtype=torch.long, device=device)
            r_t = torch.tensor([r], dtype=torch.long, device=device)

            cand_scores = []
            for s in range(0, candidates.shape[0], score_batch_size):
                ids = candidates[s : s + score_batch_size]
                t_t = torch.from_numpy(ids).to(device=device, dtype=torch.long)
                h_rep = h_t.expand(t_t.shape[0])
                r_rep = r_t.expand(t_t.shape[0])
                scores = model.score(h_rep, r_rep, t_t)
                cand_scores.append(scores.detach().cpu())
            cand_scores = torch.cat(cand_scores, dim=0)

            top_vals, top_idx = torch.topk(cand_scores, k=min(max_k, cand_scores.shape[0]), largest=True)
            top_ids = candidates[top_idx.numpy()]
            top_vals_np = top_vals.numpy()

            for k in k_values:
                shortlist = top_ids[:k]
                hit = int(np.any(shortlist == true_t))
                coverage_overall[k] += hit

                rb = rel_to_bucket.get(r, "medium")
                db = deg_bucket(int(deg[true_t]), q1, q2)
                by_rel_freq[rb][k] += hit
                by_deg[db][k] += hit

            rb = rel_to_bucket.get(r, "medium")
            db = deg_bucket(int(deg[true_t]), q1, q2)
            denom_rel[rb] += 1
            denom_deg[db] += 1

            rec = {
                "query_index": i,
                "head": h,
                "relation": r,
                "true_tail": true_t,
                "tail_type": t_type,
                "top_candidates": [
                    {"rank": int(j + 1), "tail": int(tid), "score": float(sc)}
                    for j, (tid, sc) in enumerate(zip(top_ids.tolist(), top_vals_np.tolist()))
                ],
            }
            f.write(json.dumps(rec) + "\n")

    qn = int(query_indices.shape[0])
    coverage = {f"recall@{k}": float(coverage_overall[k] / qn) for k in k_values}
    coverage_by_rel = {
        b: {f"recall@{k}": float(by_rel_freq[b][k] / max(denom_rel[b], 1)) for k in k_values}
        for b in ["rare", "medium", "frequent"]
    }
    coverage_by_deg = {
        b: {f"recall@{k}": float(by_deg[b][k] / max(denom_deg[b], 1)) for k in k_values}
        for b in ["low", "medium", "high"]
    }

    return {
        "num_queries": qn,
        "coverage_overall": coverage,
        "coverage_by_relation_frequency": coverage_by_rel,
        "coverage_by_degree": coverage_by_deg,
        "relation_bucket_sizes": denom_rel,
        "degree_bucket_sizes": denom_deg,
        "relation_counts": rel_count,
    }


def deg_size_from_split(train: Dict[str, np.ndarray], split: Dict[str, np.ndarray]) -> int:
    return int(max(train["head"].max(), train["tail"].max(), split["head"].max(), split["tail"].max()) + 1)


def parse_k_values(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = sorted(list(set([v for v in vals if v > 0])))
    if not vals:
        raise ValueError("k_values must contain positive integers")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Subset 1: Candidate generation from best Stage 1 KGE")
    parser.add_argument("--model", type=str, choices=["auto", "distmult", "complex"], default="auto")
    parser.add_argument("--distmult_metrics", type=str, default="outputs/stage1_subset2_distmult_metrics.json")
    parser.add_argument("--complex_metrics", type=str, default="outputs/stage1_subset2_complex_metrics.json")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--neg_k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--target_split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--k_values", type=str, default="50,100")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--score_batch_size", type=int, default=16384)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--checkpoint_in", type=str, default="")
    parser.add_argument("--checkpoint_out", type=str, default="outputs/stage2_subset1_best_model.pt")
    parser.add_argument("--candidates_out", type=str, default="outputs/stage2_subset1_candidates.jsonl")
    parser.add_argument("--summary_out", type=str, default="outputs/stage2_subset1_summary.json")
    args = parser.parse_args()

    model_name = args.model
    if model_name == "auto":
        model_name = auto_select_best_model(args.distmult_metrics, args.complex_metrics)
        print(f"[subset2.1] auto-selected model={model_name}")

    k_values = parse_k_values(args.k_values)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[subset2.1] device={device}")

    ctx = load_biokg()
    train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    valid = split_to_global(ctx.split_edge["valid"], ctx.type_offsets)
    test = split_to_global(ctx.split_edge["test"], ctx.type_offsets)
    target_split = valid if args.target_split == "valid" else test

    model = build_model(model_name, ctx.num_entities, ctx.num_relations, args.embedding_dim).to(device)

    train_info = {}
    if args.checkpoint_in and os.path.exists(args.checkpoint_in):
        state = torch.load(args.checkpoint_in, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        train_info = state.get("train_info", {})
        print(f"[subset2.1] loaded checkpoint: {args.checkpoint_in}")
    else:
        print("[subset2.1] training generator model...")
        train_info, best_state = train_kge(
            model,
            train,
            valid,
            ctx,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            neg_k=args.neg_k,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        model.load_state_dict(best_state)
        ckpt_dir = os.path.dirname(args.checkpoint_out)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model": model_name,
                "state_dict": best_state,
                "train_info": train_info,
                "config": {
                    "embedding_dim": args.embedding_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "neg_k": args.neg_k,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "eval_every": args.eval_every,
                    "seed": args.seed,
                },
            },
            args.checkpoint_out,
        )
        print(f"[subset2.1] saved checkpoint: {args.checkpoint_out}")

    buckets = type_id_buckets(ctx.type_offsets, ctx.num_nodes_by_type)
    summary = generate_candidates(
        model=model,
        split=target_split,
        train=train,
        type_buckets=buckets,
        k_values=k_values,
        device=device,
        score_batch_size=args.score_batch_size,
        out_path=args.candidates_out,
        max_queries=args.max_queries,
    )

    result = {
        "stage": "stage2_subset1_candidate_generation",
        "generator_model": model_name,
        "target_split": args.target_split,
        "k_values": k_values,
        "device": str(device),
        "train_info": train_info,
        "summary": summary,
        "outputs": {
            "candidates_jsonl": args.candidates_out,
            "summary_json": args.summary_out,
            "checkpoint": args.checkpoint_out if not args.checkpoint_in else args.checkpoint_in,
        },
    }

    out_dir = os.path.dirname(args.summary_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
