# AASHISH MAIN TRACK

You are the integration owner.

## Responsibilities
- Freeze the benchmark.
- Receive isolated teammate results.
- Re-run baseline before every integration.
- Integrate only one accepted change at a time.
- Run final combined benchmark.
- Maintain the final external contract.
- Maintain final submission/referee package.

## Main benchmark
`data/phase2_dev/pairs.csv`

## Integration gate
For every proposal:
1. Reproduce baseline.
2. Apply one change.
3. Re-run the same benchmark.
4. Compare Set A/B localization, weighted localization, pose, Set C F1, Spearman rho, latency, and failure taxonomy.
5. Reject any change that materially damages the overall objective.
6. Only then test combinations.

## Critical retrieval rule
A ranker cannot recover a GT candidate that was never retrieved. Always report conditional ranking performance separately from retrieval recall.
