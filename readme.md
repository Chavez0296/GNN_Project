# GNN Project - Stage 1 (Subset 1 + Subset 2 + Subset 3 + Subset 4)

This repository includes implementations for:

- Stage 1 / Subset 1: Type-aware heuristic baseline using mined length-2 metapath templates.
- Stage 1 / Subset 2: KGE baselines with type-aware negative sampling (DistMult and ComplEx).
- Stage 1 / Subset 3: R-GCN baseline with relation-collapse and negative-sampling ablations.
- Stage 1 / Subset 4: Baseline diagnostics (relation frequency slices, degree slices, and error case mining).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads `ogbl-biokg` using OGB.

## Subset 1 - Heuristic baseline

```bash
python stage1_subset1_heuristic.py
```

Output:

- `outputs/stage1_subset1_metrics.json`

## Subset 2 - KGE baselines

Run DistMult:

```bash
python stage1_subset2_kge.py --model distmult
```

Run ComplEx:

```bash
python stage1_subset2_kge.py --model complex
```

Outputs:

- `outputs/stage1_subset2_distmult_metrics.json`
- `outputs/stage1_subset2_complex_metrics.json`

## Subset 3 - Heterogeneous GNN baseline (R-GCN)

Run base R-GCN:

```bash
python stage1_subset3_gnn.py
```

Run required ablations in one call:

```bash
python stage1_subset3_gnn.py --run_ablations
```

Outputs:

- `outputs/stage1_subset3_rgcn_metrics.json` (single run or merged ablation report)
- `outputs/stage1_subset3_rgcn_base_metrics.json`
- `outputs/stage1_subset3_rgcn_ablation_homogeneous_metrics.json`
- `outputs/stage1_subset3_rgcn_ablation_globalneg_metrics.json`

## Subset 4 - Baseline diagnostics

Run diagnostics:

```bash
python stage1_subset4_diagnostics.py
```

Outputs:

- `outputs/stage1_subset4_diagnostics.json`
- `outputs/stage1_subset4_diagnostics.md`

What Subset 4 includes:

- Relation frequency slices (rare/medium/frequent) across available baseline result files.
- Degree-sliced MRR diagnostics on a sampled test subset.
- High-confidence false-positive and false-negative examples with heuristic failure labels.

## Files

- `src/biokg_data.py`: shared data loading and global ID conversion utilities.
- `stage1_subset1_heuristic.py`: metapath heuristic baseline.
- `stage1_subset2_kge.py`: DistMult and ComplEx training/evaluation.
- `stage1_subset3_gnn.py`: R-GCN baseline and ablation runner.
- `stage1_subset4_diagnostics.py`: Stage 1 diagnostics and error analysis.
- `stage2_subset1_candidate_generation.py`: Stage 2 subset 1 candidate generation.
- `requirements.txt`: dependencies.

## Stage 2 - Subset 1 (Candidate generation)

```bash
python stage2_subset1_candidate_generation.py
```

Main outputs:

- `outputs/stage2_subset1_candidates.jsonl`
- `outputs/stage2_subset1_summary.json`
- `outputs/stage2_subset1_best_model.pt`

## Stage 2 - Subset 2 (3-hop hybrid evidence retrieval)

```bash
python stage2_subset2_evidence_retrieval.py
```

Main outputs:

- `outputs/stage2_subset2_evidence.jsonl`
- `outputs/stage2_subset2_summary.json`

## Stage 2 - Subset 3 (LLM reranker with grounding)

```bash
python stage2_subset3_llm_reranker.py --evidence_in outputs/stage2_subset2_evidence.jsonl --model gemini-3.1 --max_queries 500
```

Main outputs:

- `outputs/stage2_subset3_rerank.json`
- `outputs/stage2_subset3_rerank_queries.jsonl`
