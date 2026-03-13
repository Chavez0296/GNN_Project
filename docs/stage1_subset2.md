# Stage 1 - Subset 2 (KGE Baselines)

## Final Implementation

- File: `stage1_subset2_kge.py`
- Models: DistMult and ComplEx
- Training setup: type-aware negative sampling, multi-seed evaluation, best-checkpoint by validation MRR

## What We Built

- DistMult baseline.
- ComplEx baseline.
- Type-aware negative sampling for both head and tail corruption.
- Validation-driven checkpoint selection.
- Mean metrics across seeds and per-relation test MRR outputs.

## Development Path

1. Implemented DistMult and ComplEx in one shared training/evaluation script.
2. Reused BioKG global-ID conversion utilities from `src/biokg_data.py`.
3. Added type-aware negatives to match the OGB KG protocol.
4. Added multi-seed support and JSON outputs for reproducibility.
5. Added per-relation MRR reporting because Stage 1 diagnostics required relation-level behavior.
6. Added documentation and plotting utilities after full runs were completed.
7. Wrote `outputs/subset2_analysis.md` to summarize DistMult vs ComplEx and recommend the stronger generator for Stage 2.

## Key Practical Lessons

- ComplEx outperformed DistMult overall and became the default Stage 2 generator.
- KGE baselines remained the strongest Stage 1 family in this project setup.
- Per-relation differences were large enough to motivate later slice-based analysis.
