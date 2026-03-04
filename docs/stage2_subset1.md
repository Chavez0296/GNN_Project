# Stage 2 - Subset 1 (Candidate Generation)

This subset implements top-K candidate generation for queries `(h, r, ?)` using the best Stage 1 KGE baseline.

## File

- `stage2_subset1_candidate_generation.py`

## What it does

1. Selects generator model
   - `--model auto` picks best between DistMult and ComplEx using existing Stage 1 metric JSON files.

2. Trains or loads generator
   - Trains KGE generator if no checkpoint is provided.
   - Can load from `--checkpoint_in` to skip retraining.

3. Generates top-K tails
   - For each query in target split (`valid` or `test`), scores all type-valid tail entities.
   - Exports top candidates to JSONL.

4. Reports shortlist coverage
   - Overall recall@K (true tail in top-K).
   - Recall@K by relation-frequency slices and degree slices.

## Commands

Default run (auto model selection, full test split):

```bash
python stage2_subset1_candidate_generation.py
```

Quick run for sanity check:

```bash
python stage2_subset1_candidate_generation.py --max_queries 2000 --k_values 50,100
```

Use existing checkpoint (no retraining):

```bash
python stage2_subset1_candidate_generation.py --checkpoint_in outputs/stage2_subset1_best_model.pt --max_queries 5000
```

## Outputs

- `outputs/stage2_subset1_candidates.jsonl`
  - One JSON record per query with top candidate tails and scores.
- `outputs/stage2_subset1_summary.json`
  - Coverage metrics and run metadata.
- `outputs/stage2_subset1_best_model.pt`
  - Generator checkpoint (if training was performed).

## Visualization

Generate coverage plots and summary table:

```bash
python visualize_stage2_subset1.py --input outputs/stage2_subset1_summary.json --outdir outputs/figures_stage2_subset1
```

Visualization outputs include:

- Overall recall@K curve
- Recall@K by relation-frequency slices
- Recall@K by degree slices
- Query count chart by slice
- Markdown coverage table
