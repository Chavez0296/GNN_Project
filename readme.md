# GNN Project - Stage 1 (Subset 1 + Subset 2)

This repository includes implementations for:

- Stage 1 / Subset 1: Type-aware heuristic baseline using mined length-2 metapath templates.
- Stage 1 / Subset 2: KGE baselines with type-aware negative sampling (DistMult and ComplEx).

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

## Files

- `src/biokg_data.py`: shared data loading and global ID conversion utilities.
- `stage1_subset1_heuristic.py`: metapath heuristic baseline.
- `stage1_subset2_kge.py`: DistMult and ComplEx training/evaluation.
- `requirements.txt`: dependencies.
