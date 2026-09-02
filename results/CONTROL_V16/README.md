# CONTROL_V16 Frozen Reference Directory

This directory contains the immutable reference files for the **V16 Control Pipeline** (Sequential Pose + Akhilesh Rescue Queue + CAR Fallback + V14-P1 Presence Gate).

## Contents
1. `CONTROL_COMMIT.txt`: Git commit hash at freeze point.
2. `CONTROL_METRICS.md`: Full 100-point competition scorecard breakdown.
3. `predictions.csv`: 180-pair prediction output matching competition schema (`pair_id,x,y,theta,scale,found,score`).
4. `failure_taxonomy.csv`: Error classification across all 180 cases.
5. `runtime.csv`: Per-stage and per-pair latency profiling.

> [!WARNING]
> **DO NOT OVERWRITE OR MODIFY FILES IN THIS DIRECTORY.**
> All future phases (V17 through V26) must benchmark against these exact control metrics.
