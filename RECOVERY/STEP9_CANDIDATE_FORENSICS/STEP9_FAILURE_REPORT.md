# STEP 9 — Engine-B Candidate Forensics

**Engine B** = original V25 recovered verbatim, cache-free, from image pixels.
140 present dev pairs. No training. No production change. GT used only to measure
distances. Runtime 168 s (5 workers).

Artifacts in this folder: `candidate_atlas_140.csv` (top-20 per pair, 15 fields),
`pair_classification_140.csv`, `ranking_failures.csv` (62 rows, selected-vs-GT
feature comparison), `retrieval_failures.csv` (31 rows), `gt_rank_distribution.csv`.

---

## GT recall (does a ≤ 5 px candidate exist at this depth)

| Level | GT recall / 140 |
|---|---:|
| top-1 (V25 ranker) | **43** |
| top-5 | 64 |
| top-10 | 78 |
| top-20 | 87 |
| top-50 | 92 |
| top-100 | 102 |
| top-200 (full ranked pool) | 105 |
| deep pool (NMS 600, r=3) | **106** |

## Classification

| Class | Count | Definition |
|---|---:|---|
| **R1** | **43** | GT ≤ 5 px and V25 ranker rank 1 — correct today |
| **R2** | **62** | GT ≤ 5 px and in the ranked 200-pool but rank > 1 — **ranking failure** |
| **R3** | **4** | GT ≤ 5 px only in the deeper pool — mild retrieval failure (pool cap) |
| **R4** | **31** | no ≤ 5 px candidate anywhere, even deep — retrieval failure / bad label |

---

## Answers

**A. Ranking failures: 62.** The true candidate is ≤ 5 px *and* sits in V25's
own 200-candidate ranked pool, but the V25 ranker placed a periodic replica
above it.

**B. Retrieval failures: 35** (R3 = 4 recoverable by a deeper pool, R4 = 31 with
no ≤ 5 px candidate at any depth).

**C. GT rank distribution in the ranked pool** (105 pairs where GT is present):

| rank | 1 | 2–5 | 6–10 | 11–20 | 21–50 | 51–100 | 101–200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pairs | 43 | 21 | 14 | 9 | 5 | 10 | 3 |

44 pairs have GT at rank 2–20 (a re-ranker over the top-20 can reach these);
18 at rank 21–200.

**D. GT ≤ 5 px but buried: 66** (62 in the ranked pool at rank > 1, + 4 only in
the deep pool). This is the primary recovery target.

**E. Completely absent (no ≤ 5 px candidate at any depth): 31** (R4) —
35 % Set A, 65 % Set B. Either genuinely un-retrievable by correlation
(periodic ambiguity so severe the true site never forms a peak), or the dev-set
label was not built through the dataset-prompt §5 verification gate and is
unhittable by any correct algorithm.

**F. Which features separate GT from the selected replica?** For the 62 R2
failures — *none do usefully*:

| feature | GT beats replica | note |
|---|---:|---|
| raw NCC | 46.8 % | coin flip — replica often correlates higher |
| context (combined) | 58.1 % | weak; V44/V45 already killed ~58–64 % features |
| centre distance (closer) | 59.7 % | weak; GT median 9.99 px closer to centre |
| gradient NCC | 46.8 % | coin flip |
| neighbour consistency | 43.5 % | coin flip |
| phase (lower penalty) | **6.5 %** | GT has a *worse* phase penalty (degraded true site) |
| psr | 0 % | dead — all zeros |
| family population | 1.6 % | dead — constant ~181 |
| margin | 25.8 % | replica has a *bigger* ranker margin |

The selected replica sits a median **155 px** from GT (a distant lattice site);
only 5/62 selected replicas are within 20 px.

**G. What is missing from the V25 feature space?** Every V25 feature is a
**local, small-region** measurement — "does this ~100 px patch look like the
reference." A periodic replica is locally identical to the true site, so all of
them collapse to a coin flip. What is absent:

1. **Whole-reference alignment residual** — warp the *entire* 1000×1000
   reference to the candidate pose and score full-field agreement. Replicas
   match the core but diverge on array edges, mat/strip boundaries, routing.
2. **Multi-ring structural residual** — core vs middle vs outer ring agreement;
   the replica's outer ring is wrong.
3. **Geometric / constellation consistency** — do several internal reference
   landmarks all map under *one* transform at this candidate? Replica: local
   structures match, relative geometry drifts.
4. **Candidate-vs-competitor global difference** — Δ(whole-patch), Δ(ring),
   Δ(geometry) against the best alternative, not the NCC margin.

---

## Achievable ceiling on this dev set (V25 model frozen)

- Perfect re-ranking of the current retrieval → **105/140 localized** (top-200
  recall) → localization ≈ 25–28 / 40, rejection F1 ≈ 0.6–0.75, calibration
  (AUC) ≈ 8–9, pose ≈ 19.6, efficiency 5, docs 10 → **≈ 80–83**.
- Add deeper / multi-hypothesis retrieval to convert ~10–20 of the 31 R4 pairs
  → **≈ 87–90**.
- The remaining R4 pairs are the hard wall: without them being bad labels,
  ~90 is the honest maximum on `data/phase2_dev` with the frozen V25 model.
  A dataset-prompt-§5-verified synthetic set would not contain them and would
  lift the ceiling.

STOP — no model implemented. STEP 10 (Global Alignment Discriminator) next.
