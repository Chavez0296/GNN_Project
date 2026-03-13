# Project Path Summary

This file gives a compact cross-subset view of how the project evolved.

## Stage 1

- **Subset 1:** built a type-aware heuristic metapath baseline, then added GPU-assisted scoring, progress reporting, and runtime fixes.
- **Subset 2:** built DistMult and ComplEx KGE baselines with type-aware negatives, per-relation reporting, and result visualizations.
- **Subset 3:** built an R-GCN baseline, repaired the initial underfitting training loop, added ablations and learning curves.
- **Subset 4:** built diagnostic slices and error mining, then optimized the analysis path for practical runtime.

## Stage 2

- **Subset 1:** built shortlist candidate generation from the best Stage 1 KGE and expanded coverage analysis to larger K values.
- **Subset 2:** built hybrid evidence retrieval, then added parallelization, random sampling, true-candidate metrics, 2-hop/3-hop comparison, and enriched JSONL metadata.
- **Subset 3:** built a Gemini reranker, aligned it more closely to the PDF schema/rules, then added conservative reranking and metadata enrichment after early harmful live runs.

## Where to Read the Detailed Path

- `docs/stage1_subset1.md`
- `docs/stage1_subset2.md`
- `docs/stage1_subset3.md`
- `docs/stage1_subset4.md`
- `docs/stage2_subset1.md`
- `docs/stage2_subset2.md`
- `docs/stage2_subset3.md`
- `docs/stage2_subset3_conservative_patch_rationale.md`
- `docs/stage2_subset3_metadata_patch_rationale.md`
