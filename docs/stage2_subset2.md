# Stage 2 - Subset 2 (2-Hop / 3-Hop Hybrid Evidence Retrieval)

This subset retrieves graph evidence for Stage 2 candidate tails using a **hybrid strategy**:

- top shortest paths
- plus top diverse paths

## File

- `stage2_subset2_evidence_retrieval.py`

## Inputs

- Candidate file from Stage 2 Subset 1:
  - `outputs/stage2_subset1_candidates.jsonl`

## Retrieval Strategy

For each `(head, relation, candidate_tail)`:

1. Enumerate directed paths from `head` to `candidate_tail` up to `max_hops` on the train graph (`2` or `3`).
2. Select a hybrid evidence set:
   - `top_shortest` paths by path length (then score tie-breakers)
   - `top_diverse` paths by relation-sequence/intermediate-node novelty

Default configuration:

- `max_hops=3`
- `top_shortest=2`
- `top_diverse=2`

2-hop ablation run:

```bash
python stage2_subset2_evidence_retrieval.py --max_hops 2
```

## Run

```bash
python stage2_subset2_evidence_retrieval.py
```

Example larger run settings:

```bash
python stage2_subset2_evidence_retrieval.py --candidates_per_query 100 --top_shortest 3 --top_diverse 3
```

Parallelized CPU execution (recommended):

```bash
python stage2_subset2_evidence_retrieval.py --num_workers 8 --chunk_size 8 --candidates_per_query 100
```

Representative random query sample:

```bash
python stage2_subset2_evidence_retrieval.py --max_queries 5000 --random_sample --sample_seed 42 --candidates_per_query 100 --num_workers 8 --chunk_size 8
```

Notes:

- `--num_workers 0` means auto (`cpu_count - 1`).
- Increase `chunk_size` if overhead is high; decrease it if memory usage spikes.
- `--random_sample` with `--max_queries` uses uniform reservoir sampling from the full candidate file.

## Outputs

- `outputs/stage2_subset2_evidence.jsonl`
  - One record per candidate with evidence path list, evidence IDs, and human-readable metadata:
    - `head_view`
    - `true_tail_view`
    - `candidate_tail_view`
    - `relation_view`
    - `node_views` / `relation_views` inside each path
- `outputs/stage2_subset2_summary.json`
  - Retrieval coverage and path statistics, including:
    - overall candidate path coverage
    - true-candidate path coverage
    - rank-bucket coverage (all candidates and true candidates)

## Development Path

1. Implemented initial 3-hop hybrid retrieval (top shortest + top diverse paths).
2. Observed that the first sequential full run was far too slow, so the script was patched for parallel CPU execution with chunked multiprocessing.
3. Added representative random query sampling because using the first `N` queries created biased relation coverage.
4. Added true-candidate evidence coverage metrics after realizing overall candidate coverage did not tell us whether the true answer was actually grounded.
5. Added 2-hop support as a clean retrieval ablation against the original 3-hop setup.
6. Added visualization support and direct 2-hop vs 3-hop comparison plots.
7. Enriched the JSONL artifacts with human-readable entity and relation metadata from OGB mapping files.

## Key Practical Lessons

- Parallelization was necessary for practical experimentation.
- Random sampling made diagnostics much more representative.
- True-candidate evidence coverage became the most important Stage 2 Subset 2 metric.
- 3-hop retrieved more evidence overall, but 2-hop remained useful as a cleaner ablation.

## Visualization

Generate plots for a single run:

```bash
python visualize_stage2_subset2.py --input outputs/stage2_subset2_summary.json --outdir outputs/figures_stage2_subset2
```

Generate direct 2-hop vs 3-hop comparison plots:

```bash
python visualize_stage2_subset2.py --input outputs/stage2_subset2_summary_h2.json --compare_input outputs/stage2_subset2_summary.json --outdir outputs/figures_stage2_subset2_compare
```
