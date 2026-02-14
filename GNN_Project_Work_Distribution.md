# GNN Project Work Distribution (Stage 1 Only)

This plan is restricted to **Stage 1 subsections only** from `GNN_Project.pdf`. Work is split evenly across the 4 groups: each group owns one Stage 1 subsection.

## Team Structure

- **Group 1:** Fahed, Kevin
- **Group 2:** Luis, Geo, David
- **Group 3:** Lester, Juan, Gauri
- **Group 4:** Annie, Nick, Diana

## Stage 1 Split (Even by Subsection)

- **Group 1 (25%):** Stage 1.1 - Heuristic baseline
- **Group 2 (25%):** Stage 1.2 - KGE baselines (DistMult, ComplEx)
- **Group 3 (25%):** Stage 1.3 - Heterogeneous GNN baseline (R-GCN/CompGCN)
- **Group 4 (25%):** Stage 1.4 - Baseline diagnostics

---

## Group 1 (Fahed, Kevin) - Stage 1.1 Heuristic Baseline

### Mission
Implement a strong sanity-check baseline using simple type-aware graph heuristics.

### Detailed Deliverables
1. **Metapath heuristic scorer**
   - Count length-2 and length-3 typed paths connecting head and candidate tail.
   - Prioritize meaningful biomedical patterns (e.g., drug -> protein -> disease).
   - Produce a final ranking score per candidate.

2. **Type-aware candidate handling**
   - Ensure candidate tails are restricted to the correct entity type.
   - Add safeguards against invalid type leakage during scoring.

3. **Heuristic baseline report**
   - Report Stage-1 metrics (MRR, Hits@1/3/10).
   - Include at least a few success/failure examples.
   - Note common heuristic failure modes (hub bias, sparse links).

### Member Allocation
- **Fahed**
  - Implement metapath extraction/counting logic.
  - Create relation-sequence templates and path feature set.

- **Kevin**
  - Implement ranking/scoring aggregation and evaluation runner.
  - Prepare baseline result tables and short findings summary.

---

## Group 2 (Luis, Geo, David) - Stage 1.2 KGE Baselines

### Mission
Train and tune two KGE models as the main embedding baselines.

### Detailed Deliverables
1. **DistMult baseline**
   - End-to-end training with type-aware negative sampling.
   - Validation tuning (embedding size, LR, regularization, batch size, negatives).
   - Multi-seed metrics table.

2. **ComplEx baseline**
   - End-to-end training under the same protocol for fair comparison.
   - Hyperparameter tuning and seed stability checks.
   - Comparative metrics vs DistMult.

3. **Per-relation behavior analysis**
   - Report relation-wise MRR/Hits.
   - Highlight relation types with strongest and weakest performance.

### Member Allocation
- **Luis**
  - Lead DistMult pipeline and training configuration.
  - Coordinate evaluator integration and run tracking.

- **David**
  - Lead ComplEx implementation and tuning.
  - Compare asymmetric-relation behavior vs DistMult.

- **Geo**
  - Lead per-relation breakdowns and consolidated benchmark tables.
  - Document reproducible run settings and final selected checkpoints.

---

## Group 3 (Lester, Juan, Gauri) - Stage 1.3 Heterogeneous GNN Baseline

### Mission
Implement the relational GNN baseline and run required ablations.

### Detailed Deliverables
1. **R-GCN or CompGCN baseline**
   - Build relation-aware message passing model.
   - Add triple-scoring decoder for link prediction.
   - Train/evaluate under Stage-1 protocol.

2. **Ablation: relation collapse**
   - Collapse all relation types to one type (homogeneous simulation).
   - Compare against full relation-aware model.

3. **Ablation: negative sampling scheme**
   - Alter negative sampling and measure impact.
   - Identify slices where sampling choice matters most.

4. **Baseline comparison output**
   - Compare GNN metrics against Group 2 KGE baselines.
   - Summarize strengths/weaknesses by relation characteristics.

### Member Allocation
- **Lester**
  - Lead core R-GCN/CompGCN architecture and training loop.
  - Manage checkpoints and core experiment stability.

- **Juan**
  - Lead relation-collapse ablation implementation.
  - Verify fair evaluator/decoder integration.

- **Gauri**
  - Lead negative-sampling ablation and comparative analysis.
  - Prepare final GNN baseline result tables.

---

## Group 4 (Annie, Nick, Diana) - Stage 1.4 Baseline Diagnostics

### Mission
Diagnose where Stage-1 baselines succeed/fail and generate actionable insights.

### Detailed Deliverables
1. **Relation-frequency slices**
   - Bin relation types by frequency.
   - Report baseline performance across bins.

2. **Degree slices**
   - Evaluate low-degree vs high-degree entities.
   - Quantify cold-start behavior by model type.

3. **Error-case analysis**
   - Inspect high-confidence false positives/false negatives.
   - Classify errors (hub bias, ambiguity, contradictory signals).

4. **Diagnostic summary artifact**
   - Deliver a concise report linking failure patterns to future improvement opportunities.
   - Provide reusable slice definitions for repeatable evaluation.

### Member Allocation
- **Annie**
  - Lead relation-frequency slicing scripts and visual tables.
  - Draft key findings on rare-relation behavior.

- **Nick**
  - Lead degree-based slicing and cold-start diagnostics.
  - Build comparative plots across all Stage-1 baselines.

- **Diana**
  - Lead structured error taxonomy and manual case review.
  - Produce final diagnostic narrative and recommendations.

---

## Shared Stage-1 Milestones

1. **M1:** Group 1 heuristic baseline complete.
2. **M2:** Group 2 KGE baselines complete.
3. **M3:** Group 3 GNN baseline + ablations complete.
4. **M4:** Group 4 diagnostics complete using outputs from Groups 1-3.
5. **M5:** Stage-1 combined report finalized.

## Definition of Done (Stage 1)

- Each group submits runnable code + config for its subsection.
- Metrics include MRR and Hits@1/3/10.
- Results include at least 3 seeds where applicable.
- Group 4 diagnostics include relation-frequency, degree slices, and error taxonomy.
- Final package contains one merged Stage-1 results table and subsection summaries.

## Explicit Exclusion

- **Stage 2 work is out of scope** in this plan (no candidate reranking, evidence retrieval, or LLM grounding tasks assigned).
