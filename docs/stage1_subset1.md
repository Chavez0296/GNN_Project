# Stage 1 - Subset 1 (Heuristic Baseline)

## Final Implementation

- File: `stage1_subset1_heuristic.py`
- Shared utilities: `src/biokg_data.py`
- Method: type-aware metapath counting baseline with mined relation-specific templates

## What We Built

- Type-aware global ID conversion for BioKG entities and negatives.
- Automatic mining of relation-specific 2-hop templates `(r1, r2)` from train triples.
- Weighted path-count scoring for candidate ranking.
- CPU evaluation path and optional GPU-assisted sparse scoring path.
- OGB metrics output: MRR, Hits@1/3/10, plus per-relation test MRR.

## Development Path

1. Built shared BioKG loader and type-local to global ID conversion in `src/biokg_data.py`.
2. Implemented a simple heuristic baseline around mined 2-hop relation templates.
3. Added output JSON reporting and per-relation breakdowns.
4. Added optional GPU support for sparse scoring/evaluation.
5. Added progress logs and progress bars because the first full runs appeared stalled during dataset conversion and template mining.
6. Optimized ID conversion after discovering large typed negative arrays were a hidden runtime bottleneck.
7. Preserved the heuristic as a sanity-check baseline and used its diagnostics later in Stage 1 Subset 4.

## Key Practical Lessons

- Template mining, not scoring, was the main runtime bottleneck.
- GPU helped the scoring path but did not solve the train-side mining cost.
- Progress reporting was necessary for usability on BioKG-sized runs.
