import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import Evaluator

from src.biokg_data import load_biokg, split_to_global


class DistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (self.entity(h) * self.relation(r) * self.entity(t)).sum(dim=-1)

    def score_tails(self, h: torch.Tensor, r: torch.Tensor, neg_tails: torch.Tensor) -> torch.Tensor:
        h_e = self.entity(h).unsqueeze(1)
        r_e = self.relation(r).unsqueeze(1)
        t_e = self.entity(neg_tails)
        return (h_e * r_e * t_e).sum(dim=-1)

    def score_heads(self, neg_heads: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = self.entity(neg_heads)
        r_e = self.relation(r).unsqueeze(1)
        t_e = self.entity(t).unsqueeze(1)
        return (h_e * r_e * t_e).sum(dim=-1)


class ComplEx(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.dim = dim
        self.entity_re = nn.Embedding(num_entities, dim)
        self.entity_im = nn.Embedding(num_entities, dim)
        self.rel_re = nn.Embedding(num_relations, dim)
        self.rel_im = nn.Embedding(num_relations, dim)
        for emb in [self.entity_re, self.entity_im, self.rel_re, self.rel_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr = self.entity_re(h)
        hi = self.entity_im(h)
        rr = self.rel_re(r)
        ri = self.rel_im(r)
        tr = self.entity_re(t)
        ti = self.entity_im(t)

        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)

    def score_tails(self, h: torch.Tensor, r: torch.Tensor, neg_tails: torch.Tensor) -> torch.Tensor:
        hr = self.entity_re(h).unsqueeze(1)
        hi = self.entity_im(h).unsqueeze(1)
        rr = self.rel_re(r).unsqueeze(1)
        ri = self.rel_im(r).unsqueeze(1)
        tr = self.entity_re(neg_tails)
        ti = self.entity_im(neg_tails)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)

    def score_heads(self, neg_heads: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr = self.entity_re(neg_heads)
        hi = self.entity_im(neg_heads)
        rr = self.rel_re(r).unsqueeze(1)
        ri = self.rel_im(r).unsqueeze(1)
        tr = self.entity_re(t).unsqueeze(1)
        ti = self.entity_im(t).unsqueeze(1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)


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


def run_epoch(
    model: nn.Module,
    optimizer,
    train: Dict[str, np.ndarray],
    type_offsets: Dict[str, int],
    num_nodes_by_type: Dict[str, int],
    batch_size: int,
    neg_k: int,
    device: torch.device,
    seed: int,
) -> float:
    model.train()
    n = train["head"].shape[0]
    rng_main = np.random.default_rng(seed)
    perm = rng_main.permutation(n)
    losses = []
    rng = np.random.default_rng(seed + 7)

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        h = torch.from_numpy(train["head"][idx]).to(device)
        t = torch.from_numpy(train["tail"][idx]).to(device)
        r = torch.from_numpy(train["relation"][idx]).to(device)

        tail_neg_np = sample_type_aware_negatives(
            train["tail_type"][idx],
            type_offsets,
            num_nodes_by_type,
            neg_k,
            rng,
        )
        head_neg_np = sample_type_aware_negatives(
            train["head_type"][idx],
            type_offsets,
            num_nodes_by_type,
            neg_k,
            rng,
        )

        tail_neg = torch.from_numpy(tail_neg_np).to(device)
        head_neg = torch.from_numpy(head_neg_np).to(device)

        pos_score = model.score(h, r, t)
        neg_tail_score = model.score_tails(h, r, tail_neg)
        neg_head_score = model.score_heads(head_neg, r, t)

        loss = (
            F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            + F.binary_cross_entropy_with_logits(neg_tail_score, torch.zeros_like(neg_tail_score))
            + F.binary_cross_entropy_with_logits(neg_head_score, torch.zeros_like(neg_head_score))
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model: nn.Module, split: Dict[str, np.ndarray], evaluator: Evaluator, batch_size: int, device: torch.device):
    model.eval()
    n = split["head"].shape[0]
    pos_tail_all = []
    neg_tail_all = []
    pos_head_all = []
    neg_head_all = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h = torch.from_numpy(split["head"][start:end]).to(device)
        t = torch.from_numpy(split["tail"][start:end]).to(device)
        r = torch.from_numpy(split["relation"][start:end]).to(device)
        tail_neg = torch.from_numpy(split["tail_neg"][start:end]).to(device)
        head_neg = torch.from_numpy(split["head_neg"][start:end]).to(device)

        pos_tail = model.score(h, r, t).cpu()
        neg_tail = model.score_tails(h, r, tail_neg).cpu()
        pos_head = model.score(h, r, t).cpu()
        neg_head = model.score_heads(head_neg, r, t).cpu()

        pos_tail_all.append(pos_tail)
        neg_tail_all.append(neg_tail)
        pos_head_all.append(pos_head)
        neg_head_all.append(neg_head)

    pos_tail_all = torch.cat(pos_tail_all, dim=0)
    neg_tail_all = torch.cat(neg_tail_all, dim=0)
    pos_head_all = torch.cat(pos_head_all, dim=0)
    neg_head_all = torch.cat(neg_head_all, dim=0)

    tail_res = evaluator.eval({"y_pred_pos": pos_tail_all, "y_pred_neg": neg_tail_all})
    head_res = evaluator.eval({"y_pred_pos": pos_head_all, "y_pred_neg": neg_head_all})

    def metric_mean(x):
        return float(np.asarray(x).mean())

    return {
        "mrr": float((metric_mean(tail_res["mrr_list"]) + metric_mean(head_res["mrr_list"])) / 2.0),
        "hits@1": float((metric_mean(tail_res["hits@1_list"]) + metric_mean(head_res["hits@1_list"])) / 2.0),
        "hits@3": float((metric_mean(tail_res["hits@3_list"]) + metric_mean(head_res["hits@3_list"])) / 2.0),
        "hits@10": float((metric_mean(tail_res["hits@10_list"]) + metric_mean(head_res["hits@10_list"])) / 2.0),
    }


def build_model(name: str, num_entities: int, num_relations: int, dim: int):
    if name.lower() == "distmult":
        return DistMult(num_entities, num_relations, dim)
    if name.lower() == "complex":
        return ComplEx(num_entities, num_relations, dim)
    raise ValueError(f"Unknown model: {name}")


def train_one_seed(args, ctx, seed: int) -> Tuple[dict, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    valid = split_to_global(ctx.split_edge["valid"], ctx.type_offsets)
    test = split_to_global(ctx.split_edge["test"], ctx.type_offsets)

    model = build_model(args.model, ctx.num_entities, ctx.num_relations, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(name="ogbl-biokg")

    best_valid = -1.0
    best_valid_metrics = None
    best_test_metrics = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            optimizer,
            train,
            ctx.type_offsets,
            ctx.num_nodes_by_type,
            args.batch_size,
            args.neg_k,
            device,
            seed + epoch,
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate(model, valid, evaluator, args.eval_batch_size, device)
            if valid_metrics["mrr"] > best_valid:
                best_valid = valid_metrics["mrr"]
                best_valid_metrics = valid_metrics
                best_test_metrics = evaluate(model, test, evaluator, args.eval_batch_size, device)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"seed={seed} epoch={epoch} loss={train_loss:.4f} "
                f"valid_mrr={valid_metrics['mrr']:.4f} best_valid_mrr={best_valid:.4f}"
            )

    # per-relation on test set with best model weights at end-of-training proxy
    if best_state is not None:
        model.load_state_dict(best_state)
    per_relation = {}
    rels = test["relation"]
    for r in np.unique(rels):
        idx = np.where(rels == r)[0]
        sliced = {
            "head": test["head"][idx],
            "tail": test["tail"][idx],
            "relation": test["relation"][idx],
            "head_neg": test["head_neg"][idx],
            "tail_neg": test["tail_neg"][idx],
        }
        per_relation[int(r)] = evaluate(model, sliced, evaluator, args.eval_batch_size, device)["mrr"]

    return best_valid_metrics, best_test_metrics, per_relation


def average_dicts(dicts):
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Stage 1 - Subset 2: DistMult and ComplEx baselines")
    parser.add_argument("--model", type=str, choices=["distmult", "complex"], required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--neg_k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ctx = load_biokg()
    valid_runs = []
    test_runs = []
    per_relation_runs = []

    for seed in args.seeds:
        valid_metrics, test_metrics, per_relation = train_one_seed(args, ctx, seed)
        valid_runs.append(valid_metrics)
        test_runs.append(test_metrics)
        per_relation_runs.append(per_relation)

    avg_valid = average_dicts(valid_runs)
    avg_test = average_dicts(test_runs)

    rel_ids = sorted(per_relation_runs[0].keys())
    avg_per_relation = {
        int(r): float(np.mean([run[int(r)] for run in per_relation_runs])) for r in rel_ids
    }

    payload = {
        "model": args.model,
        "config": {
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "neg_k": args.neg_k,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "seeds": args.seeds,
        },
        "valid_mean": avg_valid,
        "test_mean": avg_test,
        "test_mrr_by_relation_mean": avg_per_relation,
        "valid_by_seed": valid_runs,
        "test_by_seed": test_runs,
    }

    output = args.output or f"outputs/stage1_subset2_{args.model}_metrics.json"
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
