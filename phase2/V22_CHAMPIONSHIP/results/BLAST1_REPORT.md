# Blast 1 Retrieval Audit Report

## Results

| Variant | Top-10 (%) | Top-20 (%) | Top-50 (%) | Top-100 (%) | Top-200 (%) | Latency (ms) |
|---------|------------|------------|------------|-------------|-------------|--------------|
| R0      | 77.14      | 83.57      | 85.71      | 85.71       | 85.71       | 35.80        |
| R1      | 77.86      | 85.00      | 90.71      | 93.57       | 93.57       | 80.41        |
| R2      | 56.43      | 65.71      | 86.43      | 93.57       | 94.29       | 83.60        |
| R3      | 48.57      | 56.43      | 83.57      | 94.29       | 95.00       | 77.21        |

## Conclusion

**Winner:** R1 with 90.71% Top-50 Recall.
**R3 Performance:** R3 achieved 83.57% Top-50 recall ($\Delta$ = -2.14% vs R0).

**DECISION: KEEP**. R3 achieved 83.57% Top-50 recall, which exceeds the 78% threshold (even though it regressed slightly compared to the R0 baseline in Top-50, it met the absolute threshold).
*(Note: R1 was the best performing variant overall for Top-50)*
