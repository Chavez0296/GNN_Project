import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple, cast

from tqdm import tqdm

from src.biokg_data import load_biokg, load_name_mappings


def iter_query_groups(evidence_jsonl: str, max_queries: int):
    current_q = -1
    has_current = False
    bucket = []
    yielded = 0

    with open(evidence_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print("[subset2.3] warning: skipping malformed evidence JSONL line", flush=True)
                continue
            q = int(row["query_index"])
            if not has_current:
                current_q = q
                has_current = True

            if q != current_q:
                yield current_q, bucket
                yielded += 1
                if max_queries > 0 and yielded >= max_queries:
                    return
                current_q = q
                bucket = [row]
            else:
                bucket.append(row)

    if has_current and bucket:
        yield current_q, bucket


def build_entity_type_lookup() -> Dict[str, object]:
    ctx = load_biokg()
    ranges = []
    for t in ctx.type_names:
        start = int(ctx.type_offsets[t])
        end = start + int(ctx.num_nodes_by_type[t])
        ranges.append((start, end, t))
    entity_names, relation_names = load_name_mappings(ctx.dataset_root, ctx.type_offsets)
    return {"ranges": ranges, "entity_names": entity_names, "relation_names": relation_names}


def entity_type_name(entity_id: int, type_lookup: Dict[str, object]) -> str:
    ranges = cast(List[Tuple[int, int, str]], type_lookup["ranges"])
    for start, end, t in ranges:
        if start <= entity_id < end:
            return str(t)
    return "unknown"


def entity_display(entity_id: int, type_lookup: Dict[str, object]) -> str:
    entity_names = cast(Dict[int, str], type_lookup.get("entity_names", {}))
    name = entity_names.get(entity_id)
    etype = entity_type_name(entity_id, type_lookup)
    if name is None:
        return f"{entity_id} ({etype})"
    return f"{entity_id} [{name}] ({etype})"


def relation_display(relation_id: int, type_lookup: Dict[str, object]) -> str:
    relation_names = cast(Dict[int, str], type_lookup.get("relation_names", {}))
    name = relation_names.get(relation_id)
    if name is None:
        return f"relation_{relation_id}"
    return f"{relation_id} [{name}]"


def build_prompt(
    query_idx: int,
    candidates: List[dict],
    max_candidates_for_llm: int,
    max_paths_per_candidate: int,
    type_lookup: Dict[str, object],
) -> Tuple[str, Dict[int, set]]:
    candidates = sorted(candidates, key=lambda x: int(x["candidate_rank"]))
    candidates = candidates[:max_candidates_for_llm]

    head = int(candidates[0]["head"]) if candidates else -1
    rel = int(candidates[0]["relation"]) if candidates else -1
    head_type = entity_type_name(head, type_lookup) if head >= 0 else "unknown"
    head_disp = entity_display(head, type_lookup) if head >= 0 else "unknown"
    rel_disp = relation_display(rel, type_lookup) if rel >= 0 else "unknown"

    evidence_ids_by_tail: Dict[int, set] = defaultdict(set)
    lines = [
        "TASK",
        "Rerank candidate tails for the query (h, r, ?) using only the provided KG evidence.",
        "",
        "QUERY",
        f"(h={head_disp}, r={rel_disp}, ?)",
        f"head_type={head_type}",
        "candidate and path node names are provided where available from BioKG mappings.",
        "",
        "CANDIDATES (higher stage1_score means more likely):",
    ]

    for c in candidates:
        tail = int(c["candidate_tail"])
        rank = int(c["candidate_rank"])
        score = float(c["candidate_score"])
        tail_type = entity_type_name(tail, type_lookup)
        lines.append(f"{rank}) tail={entity_display(tail, type_lookup)} tail_type={tail_type} stage1_score={score:.6f}")

        eps = c.get("evidence_paths", [])[:max_paths_per_candidate]
        lines.append(f"Evidence bundle for candidate tail={tail}:")
        if not eps:
            lines.append("- No supporting paths found within budget")
        else:
            for i, ep in enumerate(eps, start=1):
                eid = f"P{i}"
                evidence_ids_by_tail[tail].add(eid)
                rel_seq = ep.get("relations", [])
                node_seq = ep.get("nodes", [])
                rel_seq_disp = [relation_display(int(r), type_lookup) for r in rel_seq]
                node_seq_disp = [entity_display(int(n), type_lookup) for n in node_seq]
                lines.append(
                    f"- {eid}: source={ep.get('source')}, length={ep.get('length')}, "
                    + f"nodes={node_seq_disp}, relations={rel_seq_disp}"
                )
        lines.append("")

    candidate_tail_set = [int(c["candidate_tail"]) for c in candidates]

    instruction = f"""
RULES
- Use only the evidence shown for each candidate.
- Prefer candidates with multiple independent, specific mechanistic connections.
- Penalize candidates supported only by very indirect similarity links unless evidence is strong.
- If evidence is absent or weak, lower the rank even if stage1_score is high.
- Be conservative: keep the baseline order unless the evidence clearly justifies a promotion.
- Do not make large rank changes without strong direct support.
- Only promote a candidate when the support is clearly stronger than nearby baseline alternatives.
- If uncertain, preserve the baseline order.

OUTPUT JSON SCHEMA (strict JSON only):
{{
  "reranked": [
    {{"tail": <int>, "rank": <int>, "score": <float 0..1>, "cites": ["P1", "P2"]}}
  ],
  "notes": "one short sentence about why the top choice won"
}}

Constraints:
- tails must be unique and chosen only from: {candidate_tail_set}
- include at most {max_candidates_for_llm} items
- cites must reference valid evidence IDs for that candidate
- rank starts at 1 and increases consecutively
- output valid JSON only (no markdown)
""".strip()

    prompt = "\n".join(lines) + "\n\n" + instruction
    return prompt, evidence_ids_by_tail


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        return text[s : e + 1]
    return text


def call_gemini(api_key: str, model: str, prompt: str, temperature: float, timeout_sec: int, retries: int) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model)
        + ":generateContent?key="
        + urllib.parse.quote(api_key)
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    wait = 1.0
    last_err = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            text = obj["candidates"][0]["content"]["parts"][0].get("text", "")
            return text
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(wait)
            wait = min(wait * 2.0, 8.0)
    raise RuntimeError(f"Gemini API call failed after retries: {last_err}")


def mock_rerank(candidates: List[dict], evidence_ids_by_tail: Dict[int, set], max_candidates_for_llm: int):
    candidates = sorted(candidates, key=lambda x: int(x["candidate_rank"]))[:max_candidates_for_llm]

    scored = []
    for c in candidates:
        score = float(c["candidate_score"])
        evidence_bonus = 0.1 * min(len(c.get("evidence_paths", [])), 4)
        scored.append((score + evidence_bonus, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked = []
    for s, c in scored:
        tail = int(c["candidate_tail"])
        valid_ids = sorted(list(evidence_ids_by_tail.get(tail, set())))
        cites = valid_ids[:2]
        reranked.append(
            {
                "tail": tail,
                "rank": 0,
                "score": float(max(0.0, min(1.0, 0.5 + 0.05 * len(cites)))),
                "cites": cites,
            }
        )
    for i, item in enumerate(reranked, start=1):
        item["rank"] = i
    return {"reranked": reranked, "notes": "mock_mode"}


def validate_and_finalize_ranking(
    model_output: dict,
    baseline_candidates: List[dict],
    evidence_ids_by_tail: Dict[int, set],
    conservative_mode: bool,
    preserve_top_n: int,
    max_upward_shift: int,
    min_valid_cites_for_promotion: int,
    min_llm_score_for_promotion: float,
) -> Tuple[List[int], dict]:
    baseline_candidates = sorted(baseline_candidates, key=lambda x: int(x["candidate_rank"]))
    baseline_order = [int(c["candidate_tail"]) for c in baseline_candidates]
    allowed = set(baseline_order)

    if isinstance(model_output, dict):
        ranking = model_output.get("reranked", model_output.get("ranking", []))
    else:
        ranking = []

    if not isinstance(ranking, list):
        ranking = []

    def _rank_key(item):
        try:
            return int(item.get("rank", 10**9))
        except Exception:
            return 10**9

    ranking = sorted(ranking, key=_rank_key)
    seen = set()
    reranked = []

    total_citations = 0
    valid_citations = 0
    citations_all_valid_by_item = []
    llm_items = {}

    for item in ranking:
        try:
            tail_raw = item.get("tail", item.get("candidate_tail"))
            tail = int(tail_raw)
        except Exception:
            continue
        if tail not in allowed or tail in seen:
            continue

        cited = item.get("cites", item.get("evidence_ids", []))
        if not isinstance(cited, list):
            cited = []

        tail_ids = evidence_ids_by_tail.get(tail, set())
        all_valid = True
        for eid in cited:
            total_citations += 1
            if str(eid) in tail_ids:
                valid_citations += 1
            else:
                all_valid = False
        citations_all_valid_by_item.append(all_valid if cited else True)

        llm_score = item.get("score", item.get("confidence", 0.0))
        try:
            llm_score = float(llm_score)
        except Exception:
            llm_score = 0.0

        llm_items[tail] = {
            "proposed_rank": _rank_key(item),
            "valid_cites": int(sum(1 for eid in cited if str(eid) in tail_ids)),
            "llm_score": llm_score,
        }

        reranked.append(tail)
        seen.add(tail)

    for t in baseline_order:
        if t not in seen:
            reranked.append(t)

    if conservative_mode:
        baseline_pos = {tail: i + 1 for i, tail in enumerate(baseline_order)}
        current = baseline_order[:]
        promote_order = sorted(
            [tail for tail in reranked if tail in llm_items],
            key=lambda t: (
                -llm_items[t]["valid_cites"],
                -llm_items[t]["llm_score"],
                llm_items[t]["proposed_rank"],
                baseline_pos[t],
            ),
        )

        locked_cutoff = max(0, int(preserve_top_n))
        max_shift = max(1, int(max_upward_shift))

        for tail in promote_order:
            info = llm_items[tail]
            if info["valid_cites"] < max(1, int(min_valid_cites_for_promotion)):
                continue
            if info["llm_score"] < float(min_llm_score_for_promotion):
                continue
            base_rank = baseline_pos[tail]
            if base_rank <= locked_cutoff:
                continue

            min_rank_allowed = max(locked_cutoff + 1, base_rank - max_shift)
            target_rank = max(min_rank_allowed, min(base_rank, int(info["proposed_rank"])))

            cur_idx = current.index(tail)
            desired_idx = max(locked_cutoff, target_rank - 1)
            if desired_idx < cur_idx:
                current.pop(cur_idx)
                current.insert(desired_idx, tail)

        reranked = current

    meta = {
        "cited_total": int(total_citations),
        "cited_valid": int(valid_citations),
        "citation_valid_rate": float(valid_citations / total_citations) if total_citations > 0 else 1.0,
        "all_item_citations_valid": bool(all(citations_all_valid_by_item)) if citations_all_valid_by_item else True,
        "conservative_mode": bool(conservative_mode),
        "min_valid_cites_for_promotion": int(min_valid_cites_for_promotion),
        "min_llm_score_for_promotion": float(min_llm_score_for_promotion),
    }
    return reranked, meta


def rank_metrics(true_tail: int, ordered_tails: List[int]) -> dict:
    if true_tail not in ordered_tails:
        return {"rank": None, "rr": 0.0, "hit@1": 0, "hit@3": 0, "hit@10": 0}
    rank = int(ordered_tails.index(true_tail) + 1)
    return {
        "rank": rank,
        "rr": 1.0 / rank,
        "hit@1": int(rank <= 1),
        "hit@3": int(rank <= 3),
        "hit@10": int(rank <= 10),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Subset 3: LLM reranker with grounding (Gemini)")
    parser.add_argument("--evidence_in", type=str, default="outputs/stage2_subset2_evidence.jsonl")
    parser.add_argument("--output_json", type=str, default="outputs/stage2_subset3_rerank.json")
    parser.add_argument("--output_jsonl", type=str, default="outputs/stage2_subset3_rerank_queries.jsonl")

    parser.add_argument("--model", type=str, default="gemini-3.1")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout_sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=0.0)

    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--max_candidates_for_llm", type=int, default=10)
    parser.add_argument("--max_paths_per_candidate", type=int, default=2)
    parser.add_argument("--no_conservative_mode", action="store_true")
    parser.add_argument("--preserve_top_n", type=int, default=5)
    parser.add_argument("--max_upward_shift", type=int, default=3)
    parser.add_argument("--min_valid_cites_for_promotion", type=int, default=2)
    parser.add_argument("--min_llm_score_for_promotion", type=float, default=0.65)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not args.mock and not api_key:
        raise ValueError("Missing Gemini API key. Provide --api_key or set GEMINI_API_KEY.")

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_dir2 = os.path.dirname(args.output_jsonl)
    if out_dir2:
        os.makedirs(out_dir2, exist_ok=True)

    print("[subset2.3] loading BioKG type metadata...", flush=True)
    type_lookup = build_entity_type_lookup()

    total_queries = 0
    parse_failures = 0
    total_citations = 0
    total_valid_citations = 0
    all_item_valid_count = 0

    base_rr = []
    rerank_rr = []
    base_h1 = []
    rerank_h1 = []
    base_h3 = []
    rerank_h3 = []
    base_h10 = []
    rerank_h10 = []
    true_in_candidates = 0

    print("[subset2.3] loading evidence groups...", flush=True)
    query_iter = iter_query_groups(args.evidence_in, args.max_queries)
    total_hint = args.max_queries if args.max_queries > 0 else None

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for q_idx, candidates in tqdm(query_iter, total=total_hint, desc="Gemini reranking", unit="query"):
            total_queries += 1
            candidates = sorted(candidates, key=lambda x: int(x["candidate_rank"]))
            true_tail = int(candidates[0]["true_tail"]) if candidates else -1
            entity_names = cast(Dict[int, str], type_lookup.get("entity_names", {}))
            relation_names = cast(Dict[int, str], type_lookup.get("relation_names", {}))

            baseline_order = [int(c["candidate_tail"]) for c in candidates]
            if true_tail in baseline_order:
                true_in_candidates += 1

            prompt, evidence_ids_by_tail = build_prompt(
                q_idx,
                candidates,
                max_candidates_for_llm=args.max_candidates_for_llm,
                max_paths_per_candidate=args.max_paths_per_candidate,
                type_lookup=type_lookup,
            )

            raw_text = ""
            parsed = None
            parse_ok = True

            try:
                if args.mock:
                    parsed = mock_rerank(candidates, evidence_ids_by_tail, args.max_candidates_for_llm)
                else:
                    raw_text = call_gemini(
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        timeout_sec=args.timeout_sec,
                        retries=args.retries,
                    )
                    parsed = json.loads(_extract_json_text(raw_text))
            except Exception:
                parse_ok = False
                parse_failures += 1
                parsed = {"ranking": []}

            reranked_order, citation_meta = validate_and_finalize_ranking(
                parsed,
                candidates,
                evidence_ids_by_tail,
                conservative_mode=not args.no_conservative_mode,
                preserve_top_n=args.preserve_top_n,
                max_upward_shift=args.max_upward_shift,
                min_valid_cites_for_promotion=args.min_valid_cites_for_promotion,
                min_llm_score_for_promotion=args.min_llm_score_for_promotion,
            )
            if citation_meta["all_item_citations_valid"]:
                all_item_valid_count += 1

            total_citations += citation_meta["cited_total"]
            total_valid_citations += citation_meta["cited_valid"]

            m_base = rank_metrics(true_tail, baseline_order)
            m_new = rank_metrics(true_tail, reranked_order)

            base_rr.append(m_base["rr"])
            rerank_rr.append(m_new["rr"])
            base_h1.append(m_base["hit@1"])
            rerank_h1.append(m_new["hit@1"])
            base_h3.append(m_base["hit@3"])
            rerank_h3.append(m_new["hit@3"])
            base_h10.append(m_base["hit@10"])
            rerank_h10.append(m_new["hit@10"])

            rec = {
                "query_index": int(q_idx),
                "true_tail": int(true_tail),
                "true_tail_name": entity_names.get(int(true_tail)),
                "true_tail_type": entity_type_name(int(true_tail), type_lookup),
                "head": int(candidates[0]["head"]) if candidates else -1,
                "head_name": entity_names.get(int(candidates[0]["head"])) if candidates else None,
                "head_type": entity_type_name(int(candidates[0]["head"]), type_lookup) if candidates else "unknown",
                "relation": int(candidates[0]["relation"]) if candidates else -1,
                "relation_name": relation_names.get(int(candidates[0]["relation"])) if candidates else None,
                "parse_ok": bool(parse_ok),
                "citation_valid_rate": float(citation_meta["citation_valid_rate"]),
                "baseline_rank": m_base["rank"],
                "reranked_rank": m_new["rank"],
                "baseline_order": baseline_order,
                "reranked_order": reranked_order,
                "baseline_top10": [
                    {
                        "tail": int(t),
                        "name": entity_names.get(int(t)),
                        "type": entity_type_name(int(t), type_lookup),
                    }
                    for t in baseline_order[:10]
                ],
                "reranked_top10": [
                    {
                        "tail": int(t),
                        "name": entity_names.get(int(t)),
                        "type": entity_type_name(int(t), type_lookup),
                    }
                    for t in reranked_order[:10]
                ],
            }
            fout.write(json.dumps(rec) + "\n")

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    qn = max(total_queries, 1)
    summary = {
        "stage": "stage2_subset3_llm_reranker",
        "config": {
            "evidence_in": args.evidence_in,
            "model": args.model,
            "temperature": args.temperature,
            "max_queries": args.max_queries,
            "max_candidates_for_llm": args.max_candidates_for_llm,
            "max_paths_per_candidate": args.max_paths_per_candidate,
            "conservative_mode": bool(not args.no_conservative_mode),
            "preserve_top_n": int(args.preserve_top_n),
            "max_upward_shift": int(args.max_upward_shift),
            "min_valid_cites_for_promotion": int(args.min_valid_cites_for_promotion),
            "min_llm_score_for_promotion": float(args.min_llm_score_for_promotion),
            "mock": bool(args.mock),
        },
        "stats": {
            "queries_processed": int(total_queries),
            "true_in_candidate_set": int(true_in_candidates),
            "candidate_set_true_recall": float(true_in_candidates / qn),
            "parse_failure_rate": float(parse_failures / qn),
            "citation_valid_rate_global": float(total_valid_citations / total_citations) if total_citations > 0 else 1.0,
            "all_item_citations_valid_rate": float(all_item_valid_count / qn),
            "baseline": {
                "mrr": float(sum(base_rr) / qn),
                "hits@1": float(sum(base_h1) / qn),
                "hits@3": float(sum(base_h3) / qn),
                "hits@10": float(sum(base_h10) / qn),
            },
            "reranked": {
                "mrr": float(sum(rerank_rr) / qn),
                "hits@1": float(sum(rerank_h1) / qn),
                "hits@3": float(sum(rerank_h3) / qn),
                "hits@10": float(sum(rerank_h10) / qn),
            },
        },
        "outputs": {
            "query_results_jsonl": args.output_jsonl,
            "summary_json": args.output_json,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
