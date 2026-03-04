import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import Evaluator

from src.biokg_data import load_biokg, split_to_global


def metric_mean(x) -> float:
    return float(np.asarray(x).mean())


def sample_negatives(
    type_names: np.ndarray,
    type_offsets: Dict[str, int],
    num_nodes_by_type: Dict[str, int],
    k: int,
    rng: np.random.Generator,
    num_entities: int,
    mode: str,
) -> np.ndarray:
    out = np.zeros((len(type_names), k), dtype=np.int64)
    if mode == "global_uniform":
        return rng.integers(0, num_entities, size=(len(type_names), k), endpoint=False, dtype=np.int64)

    for i, t in enumerate(type_names):
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        t = str(t)
        size = num_nodes_by_type[t]
        offset = type_offsets[t]
        out[i] = rng.integers(0, size, size=k, endpoint=False) + offset
    return out


def build_edges_by_relation(
    head: np.ndarray,
    rel: np.ndarray,
    tail: np.ndarray,
    num_relations: int,
    device: torch.device,
    collapse_relations: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    rel_used = np.zeros_like(rel) if collapse_relations else rel
    rel_count = 1 if collapse_relations else num_relations

    for r in range(rel_count):
        idx = np.where(rel_used == r)[0]
        if idx.size == 0:
            src = torch.empty(0, dtype=torch.long, device=device)
            dst = torch.empty(0, dtype=torch.long, device=device)
        else:
            src = torch.from_numpy(head[idx]).to(device=device, dtype=torch.long)
            dst = torch.from_numpy(tail[idx]).to(device=device, dtype=torch.long)
        out.append((src, dst))
    return out


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, x: torch.Tensor, edges_by_rel: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        out = self.self_linear(x)

        for r, (src, dst) in enumerate(edges_by_rel):
            if src.numel() == 0:
                continue
            msg = x[src] @ self.rel_weight[r]
            out.index_add_(0, dst, msg)

        return self.norm(out)


class RGCNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations_decoder: int,
        num_relations_encoder: int,
        dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.layers = nn.ModuleList([RGCNLayer(dim, dim, num_relations_encoder) for _ in range(num_layers)])
        self.rel_decoder = nn.Embedding(num_relations_decoder, dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.rel_decoder.weight)

    def encode(self, edges_by_rel: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        x = self.entity.weight
        for i, layer in enumerate(self.layers):
            h = layer(x, edges_by_rel)
            if i < len(self.layers) - 1:
                h = F.relu(h)
            x = x + h
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def score(self, x: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (x[h] * self.rel_decoder(r) * x[t]).sum(dim=-1)

    def score_tails(self, x: torch.Tensor, h: torch.Tensor, r: torch.Tensor, tail_neg: torch.Tensor) -> torch.Tensor:
        h_e = x[h].unsqueeze(1)
        r_e = self.rel_decoder(r).unsqueeze(1)
        t_e = x[tail_neg]
        return (h_e * r_e * t_e).sum(dim=-1)

    def score_heads(self, x: torch.Tensor, head_neg: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_e = x[head_neg]
        r_e = self.rel_decoder(r).unsqueeze(1)
        t_e = x[t].unsqueeze(1)
        return (h_e * r_e * t_e).sum(dim=-1)


def evaluate(
    model: RGCNLinkPredictor,
    split: Dict[str, np.ndarray],
    edges_by_rel: List[Tuple[torch.Tensor, torch.Tensor]],
    evaluator: Evaluator,
    batch_size: int,
    device: torch.device,
    collapse_relations: bool,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = model.encode(edges_by_rel)

        n = split["head"].shape[0]
        pos_tail_all = []
        neg_tail_all = []
        pos_head_all = []
        neg_head_all = []

        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            h = torch.from_numpy(split["head"][s:e]).to(device)
            t = torch.from_numpy(split["tail"][s:e]).to(device)
            r_np = split["relation"][s:e]
            r = torch.from_numpy(np.zeros_like(r_np) if collapse_relations else r_np).to(device)
            head_neg = torch.from_numpy(split["head_neg"][s:e]).to(device)
            tail_neg = torch.from_numpy(split["tail_neg"][s:e]).to(device)

            pos = model.score(x, h, r, t).cpu()
            neg_t = model.score_tails(x, h, r, tail_neg).cpu()
            neg_h = model.score_heads(x, head_neg, r, t).cpu()

            pos_tail_all.append(pos)
            neg_tail_all.append(neg_t)
            pos_head_all.append(pos)
            neg_head_all.append(neg_h)

        pos_tail_all = torch.cat(pos_tail_all, dim=0)
        neg_tail_all = torch.cat(neg_tail_all, dim=0)
        pos_head_all = torch.cat(pos_head_all, dim=0)
        neg_head_all = torch.cat(neg_head_all, dim=0)

        tail_res = evaluator.eval({"y_pred_pos": pos_tail_all, "y_pred_neg": neg_tail_all})
        head_res = evaluator.eval({"y_pred_pos": pos_head_all, "y_pred_neg": neg_head_all})

        return {
            "mrr": float((metric_mean(tail_res["mrr_list"]) + metric_mean(head_res["mrr_list"])) / 2.0),
            "hits@1": float((metric_mean(tail_res["hits@1_list"]) + metric_mean(head_res["hits@1_list"])) / 2.0),
            "hits@3": float((metric_mean(tail_res["hits@3_list"]) + metric_mean(head_res["hits@3_list"])) / 2.0),
            "hits@10": float((metric_mean(tail_res["hits@10_list"]) + metric_mean(head_res["hits@10_list"])) / 2.0),
        }


@dataclass
class RunConfig:
    collapse_relations: bool
    negative_sampling: str


def train_one_seed(
    args,
    seed: int,
    run_cfg: RunConfig,
    ctx,
    train: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
) -> Tuple[dict, dict, dict, List[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print(f"seed={seed} device=cuda")
    else:
        print(f"seed={seed} device=cpu")
    enc_rels = 1 if run_cfg.collapse_relations else ctx.num_relations
    dec_rels = 1 if run_cfg.collapse_relations else ctx.num_relations

    model = RGCNLinkPredictor(
        num_entities=ctx.num_entities,
        num_relations_decoder=dec_rels,
        num_relations_encoder=enc_rels,
        dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(name="ogbl-biokg")

    full_edges_eval = build_edges_by_relation(
        train["head"],
        train["relation"],
        train["tail"],
        ctx.num_relations,
        device,
        collapse_relations=run_cfg.collapse_relations,
    )

    best_valid = -1.0
    best_valid_metrics: Optional[dict] = None
    best_test_metrics: Optional[dict] = None
    best_state = None
    history: List[dict] = []

    n_train = train["head"].shape[0]
    message_edges = min(args.message_edges_per_epoch, n_train) if args.message_edges_per_epoch > 0 else n_train
    pos_per_epoch = min(args.train_pos_per_epoch, n_train) if args.train_pos_per_epoch > 0 else n_train

    for epoch in range(1, args.epochs + 1):
        edge_idx = rng.choice(n_train, size=message_edges, replace=False)
        train_edges = build_edges_by_relation(
            train["head"][edge_idx],
            train["relation"][edge_idx],
            train["tail"][edge_idx],
            ctx.num_relations,
            device,
            collapse_relations=run_cfg.collapse_relations,
        )

        step_losses = []
        steps = max(1, int(np.ceil(pos_per_epoch / args.train_batch_size)))
        for _ in range(steps):
            pos_idx = rng.choice(n_train, size=min(args.train_batch_size, n_train), replace=False)

            h = torch.from_numpy(train["head"][pos_idx]).to(device)
            t = torch.from_numpy(train["tail"][pos_idx]).to(device)
            r_np = train["relation"][pos_idx]
            r = torch.from_numpy(np.zeros_like(r_np) if run_cfg.collapse_relations else r_np).to(device)

            tail_neg_np = sample_negatives(
                train["tail_type"][pos_idx],
                ctx.type_offsets,
                ctx.num_nodes_by_type,
                args.neg_k,
                rng,
                ctx.num_entities,
                mode=run_cfg.negative_sampling,
            )
            head_neg_np = sample_negatives(
                train["head_type"][pos_idx],
                ctx.type_offsets,
                ctx.num_nodes_by_type,
                args.neg_k,
                rng,
                ctx.num_entities,
                mode=run_cfg.negative_sampling,
            )
            tail_neg = torch.from_numpy(tail_neg_np).to(device)
            head_neg = torch.from_numpy(head_neg_np).to(device)

            model.train()
            x = model.encode(train_edges)
            pos_score = model.score(x, h, r, t)
            neg_t_score = model.score_tails(x, h, r, tail_neg)
            neg_h_score = model.score_heads(x, head_neg, r, t)

            loss = (
                F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
                + F.binary_cross_entropy_with_logits(neg_t_score, torch.zeros_like(neg_t_score))
                + F.binary_cross_entropy_with_logits(neg_h_score, torch.zeros_like(neg_h_score))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_losses.append(float(loss.item()))

        avg_epoch_loss = float(np.mean(step_losses))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = evaluate(
                model,
                valid,
                full_edges_eval,
                evaluator,
                args.eval_batch_size,
                device,
                collapse_relations=run_cfg.collapse_relations,
            )
            if valid_metrics["mrr"] > best_valid:
                best_valid = valid_metrics["mrr"]
                best_valid_metrics = valid_metrics
                best_test_metrics = evaluate(
                    model,
                    test,
                    full_edges_eval,
                    evaluator,
                    args.eval_batch_size,
                    device,
                    collapse_relations=run_cfg.collapse_relations,
                )
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": avg_epoch_loss,
                    "valid_mrr": float(valid_metrics["mrr"]),
                    "valid_hits@1": float(valid_metrics["hits@1"]),
                    "valid_hits@3": float(valid_metrics["hits@3"]),
                    "valid_hits@10": float(valid_metrics["hits@10"]),
                    "best_valid_mrr_so_far": float(best_valid),
                }
            )

            gpu_info = ""
            if device.type == "cuda":
                alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
                gpu_info = f" cuda_mem_mb={alloc_mb:.0f}/{reserved_mb:.0f}"
            print(
                f"seed={seed} epoch={epoch} loss={avg_epoch_loss:.4f} "
                f"valid_mrr={valid_metrics['mrr']:.4f} best_valid_mrr={best_valid:.4f}{gpu_info}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    per_relation = {}
    for r_id in np.unique(test["relation"]):
        idx = np.where(test["relation"] == r_id)[0]
        sliced = {
            "head": test["head"][idx],
            "tail": test["tail"][idx],
            "relation": test["relation"][idx],
            "head_neg": test["head_neg"][idx],
            "tail_neg": test["tail_neg"][idx],
        }
        per_relation[int(r_id)] = evaluate(
            model,
            sliced,
            full_edges_eval,
            evaluator,
            args.eval_batch_size,
            device,
            collapse_relations=run_cfg.collapse_relations,
        )["mrr"]

    if best_valid_metrics is None or best_test_metrics is None:
        raise RuntimeError("Training did not produce validation/test metrics.")

    return best_valid_metrics, best_test_metrics, per_relation, history


def plot_learning_curves(history_by_seed: List[dict], outdir: str, prefix: str):
    os.makedirs(outdir, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    for i, h in enumerate(history_by_seed):
        epochs = [x["epoch"] for x in h["history"]]
        losses = [x["train_loss"] for x in h["history"]]
        ax1.plot(epochs, losses, marker="o", linewidth=1.2, label=f"seed={h['seed']}")
    ax1.set_title("Subset 3 Learning Curve: Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, f"{prefix}_train_loss_curve.png"), dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for i, h in enumerate(history_by_seed):
        epochs = [x["epoch"] for x in h["history"]]
        mrr = [x["valid_mrr"] for x in h["history"]]
        ax2.plot(epochs, mrr, marker="o", linewidth=1.2, label=f"seed={h['seed']}")
    ax2.set_title("Subset 3 Learning Curve: Validation MRR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MRR")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, f"{prefix}_valid_mrr_curve.png"), dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    mean_epochs = sorted({x["epoch"] for h in history_by_seed for x in h["history"]})
    for key, color in [("valid_hits@1", "#e45756"), ("valid_hits@3", "#4c78a8"), ("valid_hits@10", "#54a24b")]:
        vals = []
        for ep in mean_epochs:
            ep_vals = [x[key] for h in history_by_seed for x in h["history"] if x["epoch"] == ep]
            vals.append(float(np.mean(ep_vals)) if ep_vals else np.nan)
        ax3.plot(mean_epochs, vals, marker="o", linewidth=1.5, label=key.replace("valid_", ""), color=color)
    ax3.set_title("Subset 3 Learning Curve: Validation Hits@K (mean across seeds)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, f"{prefix}_valid_hits_curve.png"), dpi=200)
    plt.close(fig3)


def average_dicts(dicts: List[dict]) -> dict:
    keys = list(dicts[0].keys())
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def run_experiment(args, run_cfg: RunConfig, output_path: str):
    print("[subset3] loading ogbl-biokg once for this experiment...", flush=True)
    ctx = load_biokg()
    train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    valid = split_to_global(ctx.split_edge["valid"], ctx.type_offsets)
    test = split_to_global(ctx.split_edge["test"], ctx.type_offsets)

    valid_runs = []
    test_runs = []
    per_relation_runs = []
    history_by_seed = []

    for seed in args.seeds:
        v, t, rel, hist = train_one_seed(args, seed, run_cfg, ctx, train, valid, test)
        valid_runs.append(v)
        test_runs.append(t)
        per_relation_runs.append(rel)
        history_by_seed.append({"seed": int(seed), "history": hist})

    rel_ids = sorted(per_relation_runs[0].keys())
    mean_rel = {int(r): float(np.mean([x[int(r)] for x in per_relation_runs])) for r in rel_ids}

    payload = {
        "model": "rgcn_distmult_decoder",
        "config": {
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "train_pos_per_epoch": args.train_pos_per_epoch,
            "train_batch_size": args.train_batch_size,
            "message_edges_per_epoch": args.message_edges_per_epoch,
            "eval_batch_size": args.eval_batch_size,
            "neg_k": args.neg_k,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "seeds": args.seeds,
            "collapse_relations": run_cfg.collapse_relations,
            "negative_sampling": run_cfg.negative_sampling,
            "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        },
        "valid_mean": average_dicts(valid_runs),
        "test_mean": average_dicts(test_runs),
        "test_mrr_by_relation_mean": mean_rel,
        "valid_by_seed": valid_runs,
        "test_by_seed": test_runs,
        "history_by_seed": history_by_seed,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not args.no_plots:
        cfg_name = "homogeneous" if run_cfg.collapse_relations else "relational"
        neg_name = "globalneg" if run_cfg.negative_sampling == "global_uniform" else "typeaware"
        prefix = f"subset3_{cfg_name}_{neg_name}"
        fig_dir = os.path.join("outputs", "figures_subset3")
        plot_learning_curves(history_by_seed, fig_dir, prefix)

    print(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Stage 1 - Subset 3: R-GCN baseline + ablations")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_pos_per_epoch", type=int, default=131072)
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--message_edges_per_epoch", type=int, default=400000)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--neg_k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--collapse_relations", action="store_true")
    parser.add_argument("--negative_sampling", type=str, choices=["type_aware", "global_uniform"], default="type_aware")
    parser.add_argument("--run_ablations", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/stage1_subset3_rgcn_metrics.json")
    args = parser.parse_args()

    if args.run_ablations:
        base = run_experiment(
            args,
            RunConfig(collapse_relations=False, negative_sampling="type_aware"),
            "outputs/stage1_subset3_rgcn_base_metrics.json",
        )
        homogeneous = run_experiment(
            args,
            RunConfig(collapse_relations=True, negative_sampling="type_aware"),
            "outputs/stage1_subset3_rgcn_ablation_homogeneous_metrics.json",
        )
        neg_alt = run_experiment(
            args,
            RunConfig(collapse_relations=False, negative_sampling="global_uniform"),
            "outputs/stage1_subset3_rgcn_ablation_globalneg_metrics.json",
        )
        report = {
            "subset3": {
                "base": base,
                "ablation_homogeneous_relations": homogeneous,
                "ablation_global_negative_sampling": neg_alt,
            }
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    run_experiment(
        args,
        RunConfig(collapse_relations=args.collapse_relations, negative_sampling=args.negative_sampling),
        args.output,
    )


if __name__ == "__main__":
    main()
