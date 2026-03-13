# Stage 2 - Subset 3 (LLM Reranker with Grounding)

This subset reranks candidates using evidence retrieved in Stage 2 Subset 2.

## File

- `stage2_subset3_llm_reranker.py`

## Inputs

- Evidence JSONL from Subset 2:
  - `outputs/stage2_subset2_evidence.jsonl` (3-hop main)
  - or `outputs/stage2_subset2_evidence_h2.jsonl` (2-hop ablation)

## Features

- Gemini API call support (REST)
- Strict JSON output parsing
- Prompt rules aligned with Stage 2 Subset 3 PDF image:
  - only use supplied evidence
  - prefer independent mechanistic connections
  - penalize weak indirect-only support
  - lower rank if evidence is absent/weak even with high baseline score
- Output schema aligned with PDF example:
  - `{"reranked":[{"tail":...,"rank":...,"score":0..1,"cites":[...] }], "notes":"..."}`
- Prompt metadata enrichment from BioKG mapping files:
  - human-readable entity names where available
  - relation labels where available
  - entity type names retained for structure
- Citation validity checks against available evidence IDs
- Conservative rerank mode enabled by default:
  - preserve strong baseline order unless evidence clearly supports promotion
  - protect top baseline positions from harmful demotions
  - cap large upward shifts
  - block promotion of no-evidence candidates
- Baseline vs reranked candidate-set metrics:
  - MRR
  - Hits@1/3/10
- Parse failure and grounding validity statistics

## API Key

Set API key via environment variable:

```bash
set GEMINI_API_KEY=YOUR_KEY_HERE
```

Or pass directly with `--api_key`.

## Commands

3-hop reranking (main):

```bash
python stage2_subset3_llm_reranker.py --evidence_in outputs/stage2_subset2_evidence.jsonl --model gemini-3.1 --max_queries 500
```

Conservative settings are active by default. Key defaults now are:

- `max_candidates_for_llm=10`
- `max_paths_per_candidate=2`
- `preserve_top_n=5`
- `max_upward_shift=3`
- `min_valid_cites_for_promotion=2`
- `min_llm_score_for_promotion=0.65`

2-hop reranking (ablation):

```bash
python stage2_subset3_llm_reranker.py --evidence_in outputs/stage2_subset2_evidence_h2.jsonl --model gemini-3.1 --max_queries 500 --output_json outputs/stage2_subset3_rerank_h2.json --output_jsonl outputs/stage2_subset3_rerank_h2_queries.jsonl
```

Mock/offline mode for pipeline testing:

```bash
python stage2_subset3_llm_reranker.py --mock --max_queries 200
```

## Outputs

- `outputs/stage2_subset3_rerank.json`
- `outputs/stage2_subset3_rerank_queries.jsonl`

The query-level JSONL now includes human-readable metadata fields for analysis, such as:

- `head_name`, `head_type`
- `relation_name`
- `true_tail_name`, `true_tail_type`
- `baseline_top10` and `reranked_top10` with names and types

## Development Path

1. Implemented the first Gemini-based grounded reranker with structured JSON output and citation validation.
2. Verified the pipeline with mock mode, then with live Gemini calls.
3. Early live runs showed the reranker was faithful but harmful: parse reliability was good and citations were valid, yet ranking metrics dropped.
4. Re-read the PDF image and patched the prompt/output schema to better match the project rules.
5. Added a conservative rerank mode to prevent large unsupported rank changes and preserve strong baseline ordering.
6. Added progress bars and clearer runtime visibility for real API runs.
7. Added metadata enrichment so prompts and JSONL outputs include human-readable entity names and relation labels where available.
8. Preserved full grounding checks and baseline-vs-reranked metrics throughout all revisions.

## Key Practical Lessons

- Faithful reranking is not the same as beneficial reranking.
- Conservative post-processing greatly reduced damage from the LLM.
- Metadata enrichment improved interpretability and moved the system closer to the PDF design, even though gains were still limited.

## Patch Rationale

### Conservative reranker patch

The first live Gemini runs showed that the reranker was already operational and grounded:

- real Gemini API calls succeeded,
- parse failure rate was low,
- citation validity was perfect,
- evidence-grounding constraints were respected.

However, ranking quality dropped relative to the baseline generator. The initial reranker was too permissive:

- it had too much freedom to rewrite the ranking,
- evidence could be sparse or noisy,
- higher-hop paths could be weakly informative,
- the baseline generator already encoded strong latent ranking signal,
- no-evidence candidates were not blocked strongly enough,
- good grounding did not automatically imply good ranking.

The conservative patch was introduced to keep the reranker faithful to the PDF while making it safer:

1. preserve the baseline unless evidence is clearly stronger,
2. reduce catastrophic demotions of already-good baseline candidates,
3. prevent no-evidence candidates from being promoted over evidenced candidates,
4. align the prompt and JSON schema more closely with the Stage 2 PDF example image,
5. expose more human-readable structure even when names were incomplete.

What changed in that patch:

- PDF-aligned prompt sections (`TASK`, `QUERY`, candidate bundles, rules)
- PDF-style JSON schema:

```json
{
  "reranked": [
    {"tail": 123, "rank": 1, "score": 0.82, "cites": ["P1", "P2"]}
  ],
  "notes": "one short sentence"
}
```

- conservative promotion-only merge over the baseline order,
- valid-citation-only promotion eligibility,
- protected top baseline positions,
- smaller LLM context defaults.

Scientific reason for the patch:

> Can graph evidence make targeted improvements on hard cases without damaging the baseline ranking signal?

This made the reranker much safer and reduced catastrophic failures, even though it did not guarantee net gains.

### Metadata enrichment patch

The Stage 2 PDF explicitly expects the reranker to see:

- the query,
- candidate entity names,
- relation labels,
- the evidence bundle for each candidate.

Our earlier reranker only partially satisfied that requirement. It used:

- candidate IDs,
- relation IDs,
- path node IDs,
- evidence IDs,
- entity type names,
- baseline scores.

That left the LLM reasoning mostly over opaque numeric identifiers plus structure. The metadata enrichment patch fixed that using the official OGB BioKG mapping files:

- `dataset/ogbl_biokg/mapping/*_entidx2name.csv.gz`
- `dataset/ogbl_biokg/mapping/relidx2relname.csv.gz`

What was added:

1. entity display strings:
   - `global_id [mapped_name] (entity_type)`
2. relation display strings:
   - `rel_id [relation_name]`
3. readable path evidence:
   - path nodes display mapped names where available,
   - path relations display mapped labels where available.

Why this mattered:

- relation labels such as `drug-disease` are more interpretable than `relation_1`,
- symbolic names like `CID000000001` or `GO:0000002` are still far more useful than raw integers,
- richer prompt semantics give the LLM better anchors for comparison,
- the patch brought the implementation closer to the PDF design,
- it improved interpretability without weakening grounding guarantees.

This patch did not alter the candidate set, retrieval algorithm, or citation checks; it only enriched representation.

### Final tightening patch

After the conservative and metadata-enrichment patches, the reranker had become:

- stable,
- low-parse-failure,
- perfectly citation-valid,
- far less harmful than the original implementation.

But it still did not achieve a net gain over the baseline. The remaining issue was that even small weak promotions could still hurt MRR.

The final tightening patch therefore made the reranker even more selective.

Key diagnosis:

1. too many candidates were still being exposed to the LLM,
2. promotion thresholds were still too weak.

What changed:

- top-10 reranking default:
  - `max_candidates_for_llm = 10`
- stronger baseline preservation:
  - `preserve_top_n = 5`
  - `max_upward_shift = 3`
- stronger-evidence promotion rule:
  - `min_valid_cites_for_promotion = 2`
  - `min_llm_score_for_promotion = 0.65`

Why this is principled:

- the reranker should not try to outperform the generator by force,
- it should intervene only when evidence is clearly stronger than the baseline signal,
- candidate-set recall is limited,
- evidence is still partly symbolic,
- 3-hop retrieval can introduce noisy support,
- small ranking mistakes can still reduce MRR.

This patch therefore shifted the reranker from:

- "make grounded changes when possible"

to:

- "make grounded changes only when support is strong enough to justify risk."

Expected outcome:

- fewer harmful changes,
- fewer low-confidence promotions,
- safer top-of-list behavior,
- clearer measurement of whether evidence is truly strong enough to help.
