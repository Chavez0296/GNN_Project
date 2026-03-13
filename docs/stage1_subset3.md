# Stage 1 - Subset 3 (R-GCN Baseline)

This subset implements a heterogeneous GNN baseline for OGBL-BioKG using an R-GCN style encoder and a DistMult-style triple scoring decoder.

## File

- `stage1_subset3_gnn.py`

## What it does

- Loads OGBL-BioKG and converts typed local IDs into global entity IDs.
- Trains an R-GCN encoder with relation-specific message passing.
- Uses a triple scoring decoder to evaluate link prediction under the OGB protocol.
- Reports MRR and Hits@1/3/10 with multiple seeds.
- Reports per-relation test MRR.

## Required ablations

The script includes two required ablations from the project PDF:

1. Relation collapse (homogeneous simulation)
   - Set all relations to a single type in encoder/decoder.
2. Alternative negative sampling
   - Switch from type-aware negatives to global-uniform negatives.

## Commands

Run baseline only:

```bash
python stage1_subset3_gnn.py
```

Run all ablations:

```bash
python stage1_subset3_gnn.py --run_ablations
```

## Main outputs

- `outputs/stage1_subset3_rgcn_metrics.json`
- `outputs/stage1_subset3_rgcn_base_metrics.json`
- `outputs/stage1_subset3_rgcn_ablation_homogeneous_metrics.json`
- `outputs/stage1_subset3_rgcn_ablation_globalneg_metrics.json`

Learning-curve figures are also written by default to:

- `outputs/figures_subset3/subset3_relational_typeaware_train_loss_curve.png`
- `outputs/figures_subset3/subset3_relational_typeaware_valid_mrr_curve.png`
- `outputs/figures_subset3/subset3_relational_typeaware_valid_hits_curve.png`

For ablation runs, filenames use the corresponding configuration prefix.

Disable plot generation with:

```bash
python stage1_subset3_gnn.py --no_plots
```

## Notes

- The implementation uses edge and positive-triple sampling per epoch to keep memory/runtime practical on BioKG.
- The output JSON includes training configuration, per-seed metrics, means, and per-relation breakdowns.

## Development Path

1. Started with an R-GCN baseline and required ablation hooks (relation collapse and alternate negative sampling).
2. Added learning-curve tracking and figure generation for train loss and validation metrics.
3. Initial training underfit badly because the loop performed too little effective optimization per epoch.
4. Patched training to use proper mini-batch updates, larger per-epoch positive budgets, and clearer device logging.
5. Stabilized the encoder with residual connections and layer normalization after early runs produced near-random performance.
6. Re-ran the baseline and ablations to obtain meaningful Stage 1 results.
7. Used the final baseline outputs in Stage 1 Subset 4 diagnostics and Stage 2 work distribution planning.

## Key Practical Lessons

- The first naive R-GCN training loop was not competitive enough for BioKG.
- Small architecture and optimization changes mattered a lot more than expected.
- Once stabilized, the R-GCN became a valid baseline, but KGE models still remained stronger.
