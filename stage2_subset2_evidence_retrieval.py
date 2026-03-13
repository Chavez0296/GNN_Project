import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.biokg_data import load_biokg, load_name_mappings, split_to_global


_WORKER_PTR = None
_WORKER_REL = None
_WORKER_TAIL = None
_WORKER_CFG = {}
_WORKER_ENTITY_NAMES = None
_WORKER_RELATION_NAMES = None
_WORKER_TYPE_RANGES = None


@dataclass
class PathRecord:
    nodes: List[int]
    relations: List[int]
    length: int
    score: float


def load_candidates(
    path: str,
    max_queries: int,
    candidates_per_query: int,
    random_sample: bool,
    sample_seed: int,
) -> List[dict]:
    rows = []

    if random_sample and max_queries > 0:
        rng = np.random.default_rng(sample_seed)
        seen = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                obj["top_candidates"] = obj.get("top_candidates", [])[:candidates_per_query]
                seen += 1
                if len(rows) < max_queries:
                    rows.append(obj)
                else:
                    j = int(rng.integers(0, seen))
                    if j < max_queries:
                        rows[j] = obj
        rows.sort(key=lambda x: int(x.get("query_index", 0)))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_queries > 0 and i >= max_queries:
                break
            obj = json.loads(line)
            obj["top_candidates"] = obj.get("top_candidates", [])[:candidates_per_query]
            rows.append(obj)
    return rows


def build_adjacency(train: Dict[str, np.ndarray], num_entities: int):
    head = train["head"].astype(np.int64, copy=False)
    rel = train["relation"].astype(np.int64, copy=False)
    tail = train["tail"].astype(np.int64, copy=False)

    order = np.argsort(head, kind="mergesort")
    h_sorted = head[order]
    r_sorted = rel[order]
    t_sorted = tail[order]

    counts = np.bincount(h_sorted, minlength=num_entities)
    ptr = np.zeros(num_entities + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, r_sorted, t_sorted


def neighbors(node: int, ptr: np.ndarray, rel: np.ndarray, tail: np.ndarray):
    s = int(ptr[node])
    e = int(ptr[node + 1])
    return rel[s:e], tail[s:e]


def enumerate_paths_upto3(
    src: int,
    dst: int,
    ptr: np.ndarray,
    rel_arr: np.ndarray,
    tail_arr: np.ndarray,
    max_hops: int,
    max_branch: int,
    max_total_paths: int,
) -> List[PathRecord]:
    out: List[PathRecord] = []

    rel1, n1 = neighbors(src, ptr, rel_arr, tail_arr)
    if n1.shape[0] > max_branch:
        idx = np.argsort(n1)[:max_branch]
        rel1 = rel1[idx]
        n1 = n1[idx]

    # 1-hop
    for r_a, v1 in zip(rel1.tolist(), n1.tolist()):
        if v1 == dst:
            out.append(PathRecord(nodes=[src, dst], relations=[int(r_a)], length=1, score=1.0))
            if len(out) >= max_total_paths:
                return out

    # 2-hop and optional 3-hop
    for r_a, v1 in zip(rel1.tolist(), n1.tolist()):
        rel2, n2 = neighbors(int(v1), ptr, rel_arr, tail_arr)
        if n2.shape[0] > max_branch:
            idx2 = np.argsort(n2)[:max_branch]
            rel2 = rel2[idx2]
            n2 = n2[idx2]

        # 2-hop
        for r_b, v2 in zip(rel2.tolist(), n2.tolist()):
            if v2 == dst:
                out.append(
                    PathRecord(
                        nodes=[src, int(v1), dst],
                        relations=[int(r_a), int(r_b)],
                        length=2,
                        score=0.5,
                    )
                )
                if len(out) >= max_total_paths:
                    return out

        if max_hops < 3:
            continue

        # 3-hop
        for r_b, v2 in zip(rel2.tolist(), n2.tolist()):
            rel3, n3 = neighbors(int(v2), ptr, rel_arr, tail_arr)
            if n3.shape[0] > max_branch:
                idx3 = np.argsort(n3)[:max_branch]
                rel3 = rel3[idx3]
                n3 = n3[idx3]
            for r_c, v3 in zip(rel3.tolist(), n3.tolist()):
                if v3 == dst:
                    out.append(
                        PathRecord(
                            nodes=[src, int(v1), int(v2), dst],
                            relations=[int(r_a), int(r_b), int(r_c)],
                            length=3,
                            score=1.0 / 3.0,
                        )
                    )
                    if len(out) >= max_total_paths:
                        return out

    return out


def pick_hybrid_paths(paths: List[PathRecord], top_shortest: int, top_diverse: int) -> List[Tuple[PathRecord, str]]:
    if not paths:
        return []

    # Deduplicate exact paths
    uniq = {}
    for p in paths:
        key = (tuple(p.nodes), tuple(p.relations))
        if key not in uniq:
            uniq[key] = p
    paths = list(uniq.values())

    paths_sorted = sorted(paths, key=lambda p: (p.length, -p.score, tuple(p.relations), tuple(p.nodes)))

    selected: List[Tuple[PathRecord, str]] = []
    used = set()

    for p in paths_sorted[:top_shortest]:
        key = (tuple(p.nodes), tuple(p.relations))
        used.add(key)
        selected.append((p, "shortest"))

    rem = [p for p in paths_sorted if (tuple(p.nodes), tuple(p.relations)) not in used]

    seen_rel_seq = set(tuple(p.relations) for p, _ in selected)
    seen_mid_nodes = set()
    for p, _ in selected:
        for n in p.nodes[1:-1]:
            seen_mid_nodes.add(n)

    for _ in range(top_diverse):
        if not rem:
            break
        best_i = 0
        best_score = -1e9
        for i, p in enumerate(rem):
            rel_seq = tuple(p.relations)
            mids = set(p.nodes[1:-1])
            novelty_rel = 1.0 if rel_seq not in seen_rel_seq else 0.0
            novelty_mid = float(len([x for x in mids if x not in seen_mid_nodes]))
            diversity_score = novelty_rel * 3.0 + novelty_mid - 0.1 * p.length
            if diversity_score > best_score:
                best_score = diversity_score
                best_i = i

        chosen = rem.pop(best_i)
        selected.append((chosen, "diverse"))
        seen_rel_seq.add(tuple(chosen.relations))
        for n in chosen.nodes[1:-1]:
            seen_mid_nodes.add(n)

    return selected


def entity_type_name(entity_id: int, type_ranges: List[Tuple[int, int, str]]) -> str:
    for start, end, t in type_ranges:
        if start <= entity_id < end:
            return t
    return "unknown"


def entity_view(entity_id: int) -> dict:
    assert _WORKER_ENTITY_NAMES is not None and _WORKER_TYPE_RANGES is not None
    name = _WORKER_ENTITY_NAMES.get(entity_id)
    etype = entity_type_name(entity_id, _WORKER_TYPE_RANGES)
    return {"id": int(entity_id), "name": name, "type": etype}


def relation_view(relation_id: int) -> dict:
    assert _WORKER_RELATION_NAMES is not None
    return {"id": int(relation_id), "name": _WORKER_RELATION_NAMES.get(relation_id)}


def _init_worker(
    ptr: np.ndarray,
    rel_arr: np.ndarray,
    tail_arr: np.ndarray,
    entity_names: Dict[int, str],
    relation_names: Dict[int, str],
    type_ranges: List[Tuple[int, int, str]],
    max_hops: int,
    max_branch: int,
    max_total_paths: int,
    top_shortest: int,
    top_diverse: int,
):
    global _WORKER_PTR, _WORKER_REL, _WORKER_TAIL, _WORKER_CFG, _WORKER_ENTITY_NAMES, _WORKER_RELATION_NAMES, _WORKER_TYPE_RANGES
    _WORKER_PTR = ptr
    _WORKER_REL = rel_arr
    _WORKER_TAIL = tail_arr
    _WORKER_ENTITY_NAMES = entity_names
    _WORKER_RELATION_NAMES = relation_names
    _WORKER_TYPE_RANGES = type_ranges
    _WORKER_CFG = {
        "max_hops": int(max_hops),
        "max_branch": int(max_branch),
        "max_total_paths": int(max_total_paths),
        "top_shortest": int(top_shortest),
        "top_diverse": int(top_diverse),
    }


def _iter_chunks(rows: List[dict], chunk_size: int):
    for i in range(0, len(rows), chunk_size):
        yield rows[i : i + chunk_size]


def _rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51-100"


def _process_query_chunk(chunk: List[dict]):
    assert _WORKER_PTR is not None and _WORKER_REL is not None and _WORKER_TAIL is not None
    assert _WORKER_ENTITY_NAMES is not None and _WORKER_RELATION_NAMES is not None and _WORKER_TYPE_RANGES is not None

    lines: List[str] = []
    total_candidates = 0
    candidates_with_any_path = 0
    total_paths_emitted = 0
    true_candidates_total = 0
    true_candidates_with_any_path = 0
    true_paths_emitted = 0

    path_len_counter = {1: 0, 2: 0, 3: 0}
    source_counter = {"shortest": 0, "diverse": 0}
    rank_bucket_total = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    rank_bucket_with = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    true_rank_bucket_total = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    true_rank_bucket_with = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}

    for q in chunk:
        q_idx = int(q["query_index"])
        h = int(q["head"])
        r = int(q["relation"])
        true_t = int(q["true_tail"])

        for cand in q.get("top_candidates", []):
            total_candidates += 1
            c_tail = int(cand["tail"])
            c_rank = int(cand["rank"])
            c_score = float(cand["score"])
            rb = _rank_bucket(c_rank)
            rank_bucket_total[rb] += 1

            paths = enumerate_paths_upto3(
                src=h,
                dst=c_tail,
                ptr=_WORKER_PTR,
                rel_arr=_WORKER_REL,
                tail_arr=_WORKER_TAIL,
                max_hops=_WORKER_CFG["max_hops"],
                max_branch=_WORKER_CFG["max_branch"],
                max_total_paths=_WORKER_CFG["max_total_paths"],
            )

            picked = pick_hybrid_paths(
                paths,
                top_shortest=_WORKER_CFG["top_shortest"],
                top_diverse=_WORKER_CFG["top_diverse"],
            )
            if len(picked) > 0:
                candidates_with_any_path += 1
                rank_bucket_with[rb] += 1

            is_true = int(c_tail == true_t)
            if is_true:
                true_candidates_total += 1
                true_rank_bucket_total[rb] += 1
                if len(picked) > 0:
                    true_candidates_with_any_path += 1
                    true_rank_bucket_with[rb] += 1

            evidence_paths = []
            for p_i, (p, src_name) in enumerate(picked):
                evidence_id = f"q{q_idx}_c{c_rank}_p{p_i}"
                evidence_paths.append(
                    {
                        "evidence_id": evidence_id,
                        "source": src_name,
                        "nodes": p.nodes,
                        "relations": p.relations,
                        "node_views": [entity_view(int(n)) for n in p.nodes],
                        "relation_views": [relation_view(int(rid)) for rid in p.relations],
                        "length": p.length,
                        "path_score": p.score,
                    }
                )
                total_paths_emitted += 1
                path_len_counter[p.length] += 1
                source_counter[src_name] += 1

            if is_true:
                true_paths_emitted += len(evidence_paths)

            rec = {
                "query_index": q_idx,
                "head": h,
                "head_view": entity_view(h),
                "relation": r,
                "relation_view": relation_view(r),
                "true_tail": true_t,
                "true_tail_view": entity_view(true_t),
                "candidate_tail": c_tail,
                "candidate_tail_view": entity_view(c_tail),
                "candidate_rank": c_rank,
                "candidate_score": c_score,
                "is_true_tail": is_true,
                "evidence_paths": evidence_paths,
            }
            lines.append(json.dumps(rec))

    stats = {
        "total_candidates": total_candidates,
        "candidates_with_any_path": candidates_with_any_path,
        "total_paths_emitted": total_paths_emitted,
        "true_candidates_total": true_candidates_total,
        "true_candidates_with_any_path": true_candidates_with_any_path,
        "true_paths_emitted": true_paths_emitted,
        "path_len_counter": path_len_counter,
        "source_counter": source_counter,
        "rank_bucket_total": rank_bucket_total,
        "rank_bucket_with": rank_bucket_with,
        "true_rank_bucket_total": true_rank_bucket_total,
        "true_rank_bucket_with": true_rank_bucket_with,
    }
    return lines, stats


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Subset 2: 2-hop/3-hop hybrid evidence retrieval")
    parser.add_argument("--candidates_in", type=str, default="outputs/stage2_subset1_candidates.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="outputs/stage2_subset2_evidence.jsonl")
    parser.add_argument("--summary_out", type=str, default="outputs/stage2_subset2_summary.json")

    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--candidates_per_query", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--top_shortest", type=int, default=2)
    parser.add_argument("--top_diverse", type=int, default=2)
    parser.add_argument("--max_branch", type=int, default=200)
    parser.add_argument("--max_total_paths", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0, help="0=auto, 1=single-process")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--random_sample", action="store_true", help="Sample queries uniformly from full candidate file")
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_hops not in (2, 3):
        raise ValueError("Supported values for --max_hops are 2 or 3.")

    print("[subset2.2] loading dataset and building train graph adjacency...")
    ctx = load_biokg()
    train = split_to_global(ctx.split_edge["train"], ctx.type_offsets)
    ptr, rel_arr, tail_arr = build_adjacency(train, ctx.num_entities)
    entity_names, relation_names = load_name_mappings(ctx.dataset_root, ctx.type_offsets)
    type_ranges = []
    for t in ctx.type_names:
        start = int(ctx.type_offsets[t])
        end = start + int(ctx.num_nodes_by_type[t])
        type_ranges.append((start, end, str(t)))

    print("[subset2.2] loading candidates...")
    rows = load_candidates(
        args.candidates_in,
        args.max_queries,
        args.candidates_per_query,
        random_sample=args.random_sample,
        sample_seed=args.sample_seed,
    )
    print(f"[subset2.2] loaded queries={len(rows)}")

    if args.num_workers == 0:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    else:
        num_workers = max(1, int(args.num_workers))
    chunk_size = max(1, int(args.chunk_size))
    print(f"[subset2.2] parallel workers={num_workers}, chunk_size={chunk_size}")

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_candidates = 0
    candidates_with_any_path = 0
    total_paths_emitted = 0
    true_candidates_total = 0
    true_candidates_with_any_path = 0
    true_paths_emitted = 0

    path_len_counter = {1: 0, 2: 0, 3: 0}
    source_counter = {"shortest": 0, "diverse": 0}
    rank_bucket_total = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    rank_bucket_with = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    true_rank_bucket_total = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}
    true_rank_bucket_with = {"1-10": 0, "11-20": 0, "21-50": 0, "51-100": 0}

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        if num_workers == 1:
            _init_worker(
                ptr,
                rel_arr,
                tail_arr,
                entity_names,
                relation_names,
                type_ranges,
                args.max_hops,
                args.max_branch,
                args.max_total_paths,
                args.top_shortest,
                args.top_diverse,
            )
            total_chunks = int(math.ceil(len(rows) / chunk_size))
            for lines, stats in tqdm(
                map(_process_query_chunk, _iter_chunks(rows, chunk_size)),
                total=total_chunks,
                desc=f"Retrieving {args.max_hops}-hop evidence",
                unit="chunk",
            ):
                for line in lines:
                    fout.write(line + "\n")
                total_candidates += int(stats["total_candidates"])
                candidates_with_any_path += int(stats["candidates_with_any_path"])
                total_paths_emitted += int(stats["total_paths_emitted"])
                true_candidates_total += int(stats["true_candidates_total"])
                true_candidates_with_any_path += int(stats["true_candidates_with_any_path"])
                true_paths_emitted += int(stats["true_paths_emitted"])
                for k in [1, 2, 3]:
                    path_len_counter[k] += int(stats["path_len_counter"][k])
                for k in ["shortest", "diverse"]:
                    source_counter[k] += int(stats["source_counter"][k])
                for k in ["1-10", "11-20", "21-50", "51-100"]:
                    rank_bucket_total[k] += int(stats["rank_bucket_total"][k])
                    rank_bucket_with[k] += int(stats["rank_bucket_with"][k])
                    true_rank_bucket_total[k] += int(stats["true_rank_bucket_total"][k])
                    true_rank_bucket_with[k] += int(stats["true_rank_bucket_with"][k])
        else:
            total_chunks = int(math.ceil(len(rows) / chunk_size))
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(
                    ptr,
                    rel_arr,
                    tail_arr,
                    entity_names,
                    relation_names,
                    type_ranges,
                    args.max_hops,
                    args.max_branch,
                    args.max_total_paths,
                    args.top_shortest,
                    args.top_diverse,
                ),
            ) as ex:
                chunk_iter = _iter_chunks(rows, chunk_size)
                for lines, stats in tqdm(
                    ex.map(_process_query_chunk, chunk_iter, chunksize=1),
                    total=total_chunks,
                    desc=f"Retrieving {args.max_hops}-hop evidence",
                    unit="chunk",
                ):
                    for line in lines:
                        fout.write(line + "\n")
                    total_candidates += int(stats["total_candidates"])
                    candidates_with_any_path += int(stats["candidates_with_any_path"])
                    total_paths_emitted += int(stats["total_paths_emitted"])
                    true_candidates_total += int(stats["true_candidates_total"])
                    true_candidates_with_any_path += int(stats["true_candidates_with_any_path"])
                    true_paths_emitted += int(stats["true_paths_emitted"])
                    for k in [1, 2, 3]:
                        path_len_counter[k] += int(stats["path_len_counter"][k])
                    for k in ["shortest", "diverse"]:
                        source_counter[k] += int(stats["source_counter"][k])
                    for k in ["1-10", "11-20", "21-50", "51-100"]:
                        rank_bucket_total[k] += int(stats["rank_bucket_total"][k])
                        rank_bucket_with[k] += int(stats["rank_bucket_with"][k])
                        true_rank_bucket_total[k] += int(stats["true_rank_bucket_total"][k])
                        true_rank_bucket_with[k] += int(stats["true_rank_bucket_with"][k])

    candidate_rank_bucket_coverage = {
        k: float(rank_bucket_with[k] / max(rank_bucket_total[k], 1)) for k in ["1-10", "11-20", "21-50", "51-100"]
    }
    true_rank_bucket_coverage = {
        k: float(true_rank_bucket_with[k] / max(true_rank_bucket_total[k], 1))
        for k in ["1-10", "11-20", "21-50", "51-100"]
    }

    summary = {
        "stage": "stage2_subset2_evidence_retrieval",
        "config": {
            "max_queries": args.max_queries,
            "candidates_per_query": args.candidates_per_query,
            "max_hops": args.max_hops,
            "strategy": "hybrid_top_shortest_plus_top_diverse",
            "top_shortest": args.top_shortest,
            "top_diverse": args.top_diverse,
            "max_branch": args.max_branch,
            "max_total_paths": args.max_total_paths,
            "num_workers": num_workers,
            "chunk_size": chunk_size,
            "random_sample": bool(args.random_sample),
            "sample_seed": int(args.sample_seed),
            "candidates_in": args.candidates_in,
        },
        "stats": {
            "queries_processed": len(rows),
            "candidates_processed": total_candidates,
            "candidates_with_any_path": candidates_with_any_path,
            "candidate_path_coverage": float(candidates_with_any_path / max(total_candidates, 1)),
            "total_paths_emitted": total_paths_emitted,
            "avg_paths_per_candidate": float(total_paths_emitted / max(total_candidates, 1)),
            "true_candidates_total": true_candidates_total,
            "true_candidates_with_any_path": true_candidates_with_any_path,
            "true_candidate_path_coverage": float(true_candidates_with_any_path / max(true_candidates_total, 1)),
            "avg_paths_per_true_candidate": float(true_paths_emitted / max(true_candidates_total, 1)),
            "avg_paths_per_true_candidate_with_evidence": float(
                true_paths_emitted / max(true_candidates_with_any_path, 1)
            ),
            "path_length_counts": path_len_counter,
            "path_source_counts": source_counter,
            "candidate_rank_bucket_sizes": rank_bucket_total,
            "candidate_rank_bucket_coverage": candidate_rank_bucket_coverage,
            "true_rank_bucket_sizes": true_rank_bucket_total,
            "true_rank_bucket_coverage": true_rank_bucket_coverage,
        },
        "outputs": {
            "evidence_jsonl": args.output_jsonl,
            "summary_json": args.summary_out,
        },
    }

    out_dir_s = os.path.dirname(args.summary_out)
    if out_dir_s:
        os.makedirs(out_dir_s, exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
