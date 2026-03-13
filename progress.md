# Update Summary

## Current Status

- Stage 1 is complete across all four subsets.
- Stage 2 Subset 1 and Subset 2 are complete and producing usable artifacts.
- Stage 2 Subset 3 is implemented and functional, but is still the main bottleneck because reranking has not yet produced a net gain over the baseline.

## Stage 1 Summary

### Subset 1 - Heuristic baseline

- Implemented a type-aware heuristic baseline using mined metapath templates.
- Added progress reporting and optional GPU-assisted scoring because BioKG runtime was heavy.
- This baseline served as a sanity check and gave useful failure diagnostics, but it was not the strongest overall model.

Related artifacts:

- `outputs/stage1_subset1_metrics.json`
- `outputs/figures_subset1/subset1_overall_valid_vs_test.png`
- `outputs/figures_subset1/subset1_test_mrr_sorted.png`

### Subset 2 - KGE baselines

- Implemented DistMult and ComplEx with type-aware negative sampling.
- Added multi-seed evaluation and per-relation analysis.
- ComplEx was the strongest Stage 1 model and became the default Stage 2 generator.

Related artifacts:

- `outputs/stage1_subset2_distmult_metrics.json`
- `outputs/stage1_subset2_complex_metrics.json`
- `outputs/figures_subset2_distmult/subset2_distmult_overall.png`
- `outputs/figures_subset2_complex/subset2_complex_overall.png`
- `outputs/subset2_analysis.md`

### Subset 3 - R-GCN baseline

- Implemented an R-GCN baseline plus the required ablations.
- The first training loop underfit badly, so training was redesigned with proper minibatching and stabilization changes.
- Final R-GCN results became meaningful, but still remained below the KGE baselines.

Related artifacts:

- `outputs/stage1_subset3_rgcn_metrics.json`
- `outputs/stage1_subset3_rgcn_base_metrics.json`
- `outputs/figures_subset3/subset3_relational_typeaware_valid_mrr_curve.png`

### Subset 4 - Diagnostics

- Implemented relation-frequency slices, degree slices, and structured error mining.
- These diagnostics clearly showed that rare relations, low-degree entities, hub bias, and missing supporting paths were major issues.
- Those findings directly shaped Stage 2 design decisions.

Related artifacts:

- `outputs/stage1_subset4_diagnostics.json`
- `outputs/stage1_subset4_diagnostics.md`

## Stage 2 Summary

### Subset 1 - Candidate generation

- Built candidate generation from the strongest Stage 1 KGE baseline.
- Added recall@K analysis and expanded from smaller K to larger shortlist settings after finding that the true answer was often missing from small candidate sets.
- This subset established that larger K values are important for giving reranking enough headroom.

Main finding:

- Candidate recall improves substantially as K increases, which is why later Stage 2 steps depend heavily on shortlist size.

Related artifacts:

- `outputs/stage2_subset1_summary.json`
- `outputs/stage2_subset1_analysis.md`
- `outputs/figures_stage2_subset1/stage2_subset1_overall_recall_curve.png`
- `outputs/figures_stage2_subset1/stage2_subset1_recall_by_degree_slice.png`
- `outputs/figures_stage2_subset1/stage2_subset1_incremental_recall_gains.png`

### Subset 2 - Evidence retrieval

- Built 3-hop hybrid evidence retrieval (top shortest + top diverse).
- Added parallel CPU execution after discovering that the first full run was too slow.
- Added representative random sampling because taking the first `N` queries biased the evaluation.
- Added true-candidate evidence coverage metrics because overall candidate coverage alone was not sufficient.
- Added 2-hop as an ablation and later enriched the output JSONL with human-readable metadata.

Main findings:

- 3-hop gives better coverage and richer evidence than 2-hop.
- 2-hop is cleaner and useful as an ablation, but 3-hop remains the stronger retrieval setting overall.
- Retrieval is now one of the stronger components of Stage 2.

Related artifacts:

- `outputs/stage2_subset2_summary.json`
- `outputs/stage2_subset2_summary_h2.json`
- `outputs/figures_stage2_subset2_compare/stage2_subset2_compare_key_metrics.png`
- `outputs/figures_stage2_subset2_compare/stage2_subset2_rank_bucket_coverage.png`

### Subset 3 - LLM reranker with grounding

- Implemented a Gemini-based reranker with JSON output parsing and strict citation validation.
- Early live runs showed that the reranker was operational and faithful, but harmful to ranking quality.
- In response, multiple patches were applied:
  - a PDF-aligned schema/prompt patch,
  - a conservative reranker patch,
  - a metadata enrichment patch using BioKG mappings,
  - a final tightening patch for smaller top-of-list reranking and stronger evidence-based promotion thresholds.

Main findings:

- The reranker is now stable, grounded, interpretable, and close to neutral.
- It is no longer causing large damage, but it still does not beat the baseline ranking.
- This means the end-to-end system is working, but the final reranking stage remains the weakest link.

Related artifacts:

- `outputs/stage2_subset3_rerank.json`
- `outputs/stage2_subset3_rerank_h3_metadata.json`
- `outputs/stage2_subset3_rerank_h3_enriched.json`
- `docs/stage2_subset3.md`

## Overall Interpretation

- The project is now end-to-end functional.
- Stage 1 baselines are established and well understood.
- Stage 2 candidate generation and evidence retrieval are strong enough to support research experiments.
- The main unresolved challenge is Stage 2 Subset 3: the LLM reranker is faithful and stable, but has not yet produced consistent performance gains over the baseline generator.

## Short Takeaway

- We have a working two-stage pipeline.
- Baseline ranking and evidence retrieval are in good shape.
- The LLM reranker is the current bottleneck: it is explainable and reliable, but not yet better than the baseline ordering.
