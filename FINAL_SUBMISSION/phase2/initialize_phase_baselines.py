import os

def initialize_files():
    phases = {
        "PHASE_10": {
            "owner": "Sai Dharshan",
            "branch": "sai-dharshan-v106-v109",
            "task": "V10.6 High-Recall Candidate Retrieval",
            "baseline": """# Phase 10 Baseline Metrics

**Owner:** Sai Dharshan
**Branch:** `sai-dharshan-v106-v109`

## Retrieval Recall Baseline (Dev Set)
- Top-1 Recall: 24.29%
- Top-3 Recall: 32.86%
- Top-5 Recall: 34.29%
- Top-10 Recall: 38.57%
- Top-20 Recall: 39.29%
- Top-50 Recall: 45.71%
- Top-100 Recall: 49.29%
- Anywhere Recall: 100.00%

## System Metrics
- Set A <= 5px: 25.00%
- Set B <= 5px: 27.03%
- Weighted Loc: 26.11%
- Mean Latency: ~2.55s per pair
"""
        },
        "PHASE_11": {
            "owner": "Akhilesh",
            "branch": "akhilesh-v107-v110",
            "task": "V10.7 Replica-Aware Ranker",
            "baseline": """# Phase 11 Baseline Metrics

**Owner:** Akhilesh
**Branch:** `akhilesh-v107-v110`

## Ranking Performance Baseline
- Top-1 given GT retrieved: 61.8% (34 / 55)
- Top-3 given GT retrieved: 83.6% (46 / 55)
- Top-5 given GT retrieved: 87.3% (48 / 55)

## Overall System Metrics
- Weighted Localization: 26.11%
- Spearman rho: 0.2554
- Mean Latency: ~2.55s per pair
"""
        },
        "PHASE_12": {
            "owner": "Shanganidhi",
            "branch": "shanganidhi-v108-v111-v112",
            "task": "V10.8 Presence / Absence Decision",
            "baseline": """# Phase 12 Baseline Metrics

**Owner:** Shanganidhi
**Branch:** `shanganidhi-v108-v111-v112`

## Presence Rejection Baseline
- Set C Rejection F1: 0.1928
- Set C Precision: 0.1860
- Set C Recall: 0.2000
- Same-Architecture Rejection F1: 0.0000 (0% on Adversarial category 8)
- Cross-Architecture Rejection F1: 1.0000 (100% on Adversarial category 9)
"""
        },
        "PHASE_13": {
            "owner": "Sai Dharshan",
            "branch": "sai-dharshan-v106-v109",
            "task": "V10.9 Hard-Negative Mining",
            "baseline": """# Phase 13 Baseline Metrics

**Owner:** Sai Dharshan
**Branch:** `sai-dharshan-v106-v109`

## Mining Base
- Source: results/phase2/candidate_features.csv (3,600 candidate instances)
- Focus: GT vs FFT #1, GT vs Nearest Replica pairwise feature deltas.
"""
        },
        "PHASE_14": {
            "owner": "Akhilesh",
            "branch": "akhilesh-v107-v110",
            "task": "V10.10 Pose-Robust Retrieval",
            "baseline": """# Phase 14 Baseline Metrics

**Owner:** Akhilesh
**Branch:** `akhilesh-v107-v110`

## Pose Retrieval Baseline
- Set A Scale MAE: 0.0466
- Set A Rotation MAE: 0.0989
- Set B Scale MAE: 0.0639
- Set B Rotation MAE: 0.1813
"""
        },
        "PHASE_15": {
            "owner": "Shanganidhi",
            "branch": "shanganidhi-v108-v111-v112",
            "task": "V10.11 Calibration / Confidence Gate",
            "baseline": """# Phase 15 Baseline Metrics

**Owner:** Shanganidhi
**Branch:** `shanganidhi-v108-v111-v112`

## Calibration Baseline
- Spearman Rank Correlation (rho): 0.2554
- Calibration strategy: Piecewise linear calibration in calibration.py
"""
        },
        "PHASE_16": {
            "owner": "Shanganidhi",
            "branch": "shanganidhi-v108-v111-v112",
            "task": "V10.12 Independent Referee Validation",
            "baseline": """# Phase 16 Baseline Metrics

**Owner:** Shanganidhi
**Branch:** `shanganidhi-v108-v111-v112`

## Referee Validation Baseline
- Denominator: 180 Dev Pairs (70 Set A, 70 Set B, 40 Set C)
- Baseline weighted localization: 26.11%
- Baseline Rejection F1: 0.1928
- Baseline Spearman rho: 0.2554
"""
        }
    }
    
    for phase, info in phases.items():
        os.makedirs(phase, exist_ok=True)
        
        # Write BASELINE.md
        with open(os.path.join(phase, "BASELINE.md"), "w") as f:
            f.write(info["baseline"])
            
        # Write TASK_STATE.md
        task_state = f"""# Task State — {info["task"]}

**Owner:** {info["owner"]}
**Branch:** `{info["branch"]}`
**Status:** INITIALIZED

## Checklist
- [ ] Reconnaissance and baseline validation
- [ ] Formulation of target hypothesis
- [ ] Implementation of isolated proposed scripts
- [ ] Evaluation on authoritative main dataset
- [ ] Generation of handoff package for Aashish
"""
        with open(os.path.join(phase, "TASK_STATE.md"), "w") as f:
            f.write(task_state)
            
        print(f"Initialized BASELINE.md and TASK_STATE.md in {phase}")

if __name__ == "__main__":
    initialize_files()
