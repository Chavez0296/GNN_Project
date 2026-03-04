# Stage 1 - Subset 4 Diagnostics

## Relation Frequency Slices

| Model | Rare | Medium | Frequent |
|-------|-----:|-------:|---------:|
| subset1_heuristic | 0.3498 | 0.4486 | 0.5399 |
| subset2_distmult | 0.4905 | 0.5504 | 0.7188 |
| subset2_complex | 0.5012 | 0.5664 | 0.7358 |
| subset3_rgcn | 0.4309 | 0.5080 | 0.5768 |

## Degree Slices (heuristic diagnostic run)

- Low degree MRR: 0.003937757479486093
- Medium degree MRR: 0.008813090916301046
- High degree MRR: 0.16165814104040196

## Error Case Snapshots

- Collected high-confidence false positives: 25
- Collected high-confidence false negatives: 25

The full structured error records are available in the JSON output.