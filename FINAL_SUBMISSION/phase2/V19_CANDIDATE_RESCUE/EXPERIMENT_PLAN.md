# Phase V19: Experiment Plan

## Protocol
1. **Target**: 140 Present Cases in `data/phase2_dev/pairs.csv`. Focus specifically on the 18 cases classified as `RETRIEVAL_CAPACITY_SUPPRESSION`.
2. **Variants Tested**:
   - **R0 (V14 Baseline)**: Greedy NMS ($r=5, K=50$).
   - **R1 (V16 Control)**: Bounded Context Rescue Queue ($K=200 \to \text{Context Filter} \to \text{Top-50}$).
   - **R2 (V19 Diverse Family Compression)**: Extract $K=200$, cluster periodic families, take top representatives per family, fill remainder by Center-Context score.
   - **R3 (V19 Dual Queue + Spatial Quad Partitioning)**: Divide search FOV into central core ($r \le 250\text{px}$) and peripheral sectors, allocating 35 slots to center and 15 to periphery.
3. **Metrics**:
   - `Effective Top-50 GT Recall (%)`
   - `Effective Top-100 GT Recall (%)`
   - `Recovery of the 18 Target Cases (Count)`
   - `Extraction Latency (ms)`
