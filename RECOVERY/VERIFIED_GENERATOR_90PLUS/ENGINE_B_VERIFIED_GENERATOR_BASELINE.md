# EXP001 — Engine B baseline on the verified generator (pilot-100)

**Status:** complete. Baseline established. No promotion (measurement only).
**Data:** `experiments/EXP001_pilot100/data` — 95 shipped pairs (78 present, 17 absent), 5 dropped by the verifier.
**Engine:** Engine B = original V25 recovered verbatim, cache-free (`RECOVERY/V25_ORIGINAL`, hashes pinned in `RECOVERY/V25_ORIGINAL_SHA256.txt`).
**Runner:** `run_engineB_on_dataset.py`. No cache, no `pair_id` logic, no network, no historical predictions.

## 1. The benchmark is sound

Every shipped present label was checked by `generator/verify_ground_truth.py`, an
**independent** verifier: it sees only the re-read PNGs and the declared GT, and it
builds its template by a deliberately different path (box blur → `getRotationMatrix2D`/
`warpAffine` → `INTER_AREA`) from the generator's supersampled area integration.

| verifier metric | result over 78 present |
|---|---|
| global correlation peak error vs label | mean **0.613 px**, max 1.314 px |
| GT-vs-competitor margin | mean 0.187, min 0.121 (5th pct 0.126) |
| NCC at GT | mean 0.942 |
| GT NMS rank | **1 for 78/78** |
| recoverable ≤5 px / ≤1 px | 78/78 / 70/78 |

Coverage: 44 DRAM / 51 FinFET; 49 nominal / 25 periodic / 21 degraded; severity {0:34, 1:40, 3:10, 4:11}.

This is the property `data/phase2_dev` lacks. There, 31 of 140 present labels are not
retrievable by any code path at any pool depth; here the number is **0**.

## 2. Engine B result

```json
{"n_pairs":95,"n_present":78,"n_absent":17,
 "recall":{"top1":42,"top5":50,"top10":55,"top20":61,"top50":75,"top100":78,"top200":78},
 "deep_pool":78,"classes":{"R1":42,"R2":36,"ABSENT":17},
 "localized_le5px":30,"localized_le2px":29,"localized_le1px":29,
 "runtime_median_s":5.484,"runtime_max_s":6.522}
```

## 3. What this decomposes to

**Retrieval is solved. Ranking is the entire bottleneck.**

- **R4 (GT never retrieved) = 0/78.** On `phase2_dev` it was 31/140. The candidate
  extractor is not the problem and never was — the old number was an artefact of
  unverifiable labels.
- **R2 (GT retrieved, wrong candidate selected) = 36/78.** Every one of these is a
  pair where the correct site is sitting in the pool and the learned ranker demotes it.
- Localization credit ≤5 px is 30/78 — *below* even top-1 recall (42), because the
  presence gate additionally rejects some correctly-ranked present pairs.

## 4. Runtime is over budget

Median **5.48 s/pair** against a 5 s median budget (20 s hard timeout). Not fatal —
the efficiency component is 5 points and the reference machine differs — but it must
be brought under 5 s before any submission claim. Logged as an open item, not fixed here.

## 5. Honest limitations

- These are **synthetic** pairs from a procedural field generator. They are verified,
  which the organizer dev set is not, but they are not the organizer's distribution.
  Every conclusion here is a conclusion about *verifiable* SEM-like data, and must be
  re-checked against `data/phase2_dev` as an untouched external diagnostic.
- 95 pairs is a pilot. Effect sizes below ~5 pairs are not resolvable at this n.
- No model was fitted, tuned, or selected on any organizer data in producing this.

## 6. Decision taken

Per the mission's branch rule — *"IF R2 dominates: improve candidate discrimination"* —
R2 = 36 with R4 = 0 makes the next experiment unambiguous. See `EXP002`.
