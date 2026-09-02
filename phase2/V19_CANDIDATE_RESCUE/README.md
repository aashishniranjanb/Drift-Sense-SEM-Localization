# Phase V19: Candidate Rescue 2.0 (Aashish Main Track)

## 1. Mission Overview
Target the **18 `RETRIEVAL_CAPACITY_SUPPRESSION` failures** identified in Phase V17.
In these cases, the Ground Truth peak exists in the raw correlation plane (ranks 51–200), but standard NMS truncates them because repetitive periodic clones consume the Top-50 quota.

**Primary Goal:** Build a deterministic **Two-Queue Extraction Architecture** (Normal Queue + Intelligent Rescue Queue) that increases effective candidate pool recall without evaluating 500 candidates brute-force.

---

## 2. Architecture: Bounded Cluster-Representative Rescue
```text
                 CORRELATION PLANE
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   PRIMARY NMS                      RESCUE QUEUE
   (Top-30 Highest NCC)             (Extract 200 raw peaks)
        │                                 │
        │                        Spatial Graph Clustering
        │                        (Group periodic clones)
        │                                 │
        │                        Select 1-2 Best Reps
        │                        per Family + Center Prior
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                 COMBINED TOP-50
```

---

## 3. Key Constraints
- Runtime must remain within the $\le 5.0\text{s/pair}$ median budget.
- Top-50 candidate pool size must remain bounded at $K \le 50$.
- Must not degrade nominal Set A performance.
