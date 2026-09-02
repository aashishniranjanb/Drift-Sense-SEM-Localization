# Phase V19: Candidate Rescue 2.0 Benchmark Ablation

**Total Present Cases:** 140 | **Target Suppressed Failures:** 18

## 1. Candidate Pool Recall Comparison (Top-50 Cap)

| Extractor Architecture | Total GT Recall (%) | Set A Recall (%) | Set B Recall (%) | Target Failures Rescued | Delta vs Control |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `V14_Greedy_NMS_50` | **50.00%** (70/140) | 55.71% | 44.29% | **1 / 18** | -0.71% |
| `V16_Context_Rescue_50` | **50.71%** (71/140) | 55.71% | 45.71% | **0 / 18** | +0.00% |
| `V19_Dual_Queue_50` | **67.14%** (94/140) | 70.00% | 64.29% | **10 / 18** | +16.43% |
