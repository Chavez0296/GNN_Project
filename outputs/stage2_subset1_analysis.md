# Stage 2 - Subset 1 Candidate Generation Analysis

## Run Summary

- Script: `stage2_subset1_candidate_generation.py`
- Generator model: `ComplEx`
- Target split: `test`
- K values: `50, 100, 200, 500`
- Device: `cuda`
- Queries processed: `162,870`
- Checkpoint: `outputs/stage2_subset1_best_model.pt`
- Candidate output: `outputs/stage2_subset1_candidates.jsonl`
- Summary file: `outputs/stage2_subset1_summary.json`

## Training Behavior (Generator)

- Training converged smoothly:
  - Loss decreased from `0.5472` (epoch 2) to `0.2679` (epoch 30).
  - Proxy validation MRR increased from `0.1744` to a best `0.5984` at epoch 24.
- Slight post-peak drift from epoch 24 to 30 (`0.5984 -> 0.5962`) indicates minor late-epoch overtraining, but overall stability is good.

## Overall Candidate Coverage

Coverage is measured as the fraction of queries where the true tail appears in top-K shortlist.

| K | Recall@K | Hits (approx) | Misses (approx) |
|---|---:|---:|---:|
| 50  | 0.3012 | 49,057  | 113,813 |
| 100 | 0.4683 | 76,277  | 86,593  |
| 200 | 0.6569 | 106,987 | 55,883  |
| 500 | 0.8191 | 133,400 | 29,470  |

### Marginal gain by increasing K

- `50 -> 100`: `+0.1671`
- `100 -> 200`: `+0.1886` (largest gain segment)
- `200 -> 500`: `+0.1622`

Interpretation:

- `K=50` and `K=100` are likely too restrictive for strong reranking gains.
- `K=200` is a practical turning point.
- `K=500` provides high ceiling for reranker effectiveness.

## Relation-Frequency Slice Analysis

Relation buckets are split evenly by relation count (rare/medium/frequent relation IDs), but query volume is highly skewed to frequent relations.

- Bucket sizes in this test set:
  - Rare: `1,029` queries (`0.63%`)
  - Medium: `14,221` queries (`8.73%`)
  - Frequent: `147,620` queries (`90.64%`)

Coverage by bucket:

| Slice | R@50 | R@100 | R@200 | R@500 |
|---|---:|---:|---:|---:|
| Rare     | 0.4334 | 0.6443 | 0.8494 | 0.9495 |
| Medium   | 0.2908 | 0.4782 | 0.7518 | 0.9940 |
| Frequent | 0.3013 | 0.4661 | 0.6464 | 0.8013 |

Interpretation:

- Rare/medium relations show very high coverage at larger K.
- Frequent relations are harder and dominate aggregate performance due to volume.

## Degree Slice Analysis

Degree buckets are based on training graph degree of the true tail entity.

- Bucket sizes:
  - Low degree: `3,211` queries (`1.97%`)
  - Medium degree: `15,624` queries (`9.59%`)
  - High degree: `144,035` queries (`88.44%`)

Coverage by degree:

| Slice | R@50 | R@100 | R@200 | R@500 |
|---|---:|---:|---:|---:|
| Low    | 0.0439 | 0.0897 | 0.1426 | 0.2149 |
| Medium | 0.1386 | 0.2014 | 0.2613 | 0.3574 |
| High   | 0.3246 | 0.5057 | 0.7113 | 0.8826 |

Interpretation:

- Low-degree/cold-start entities remain the dominant failure region.
- High-degree entities have strong candidate recall, especially at `K>=200`.

## Readiness for Stage 2 Subset 2 (Evidence Retrieval)

The candidate generator is valid and ready for downstream Stage 2 use, with K selection trade-offs:

- **K=200 (recommended default):**
  - Good balance of quality and downstream retrieval/LLM cost.
  - Recall@200 = `0.6569`.

- **K=500 (upper-bound setting):**
  - Stronger ceiling for reranking gains.
  - Recall@500 = `0.8191`.
  - Higher retrieval and prompt cost.

## Practical Recommendation

Run Subset 2 with two tracks:

1. **Primary run:** `K=200` for cost-effective pipeline.
2. **Upper-bound run:** `K=500` to measure maximum possible reranking uplift.

This dual setup will make Stage 2 conclusions stronger by showing both practical and best-case behavior.
