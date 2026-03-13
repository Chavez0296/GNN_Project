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

## Development Path

1. Chose the strongest Stage 1 KGE family as the Stage 2 generator base.
2. Added automatic model selection between DistMult and ComplEx using Stage 1 metrics.
3. Implemented shortlist generation over all type-valid tails with JSONL export.
4. Added recall@K reporting, first at `50/100`, then expanded to `50/100/200/500` after coverage analysis showed larger K was necessary.
5. Added slice-aware coverage reporting by relation frequency and degree because aggregate recall hid cold-start weaknesses.
6. Added dedicated visualization and written analysis outputs to support K selection for downstream reranking.

## Key Practical Lessons

- `K=50` and `K=100` were too restrictive for later reranking.
- `K=200` became the practical default and `K=500` the upper-bound setting.
- Low-degree entities remained the largest candidate-generation weakness even with a strong generator.

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
