# V14-R2 Replica Ranking Experiment

| Metric | V14 Baseline | V14-R2 (Context-128 Ambiguity Filter) | Delta | Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Weighted Loc ($\le 5\text{ px}$)** | **48.88%** | **8.43%** | **-40.45%** | RETAIN V14 BASELINE |
| **Set A $\le 5\text{ px}$** | 38.78% | 10.00% | -28.78% | — |
| **Set B $\le 5\text{ px}$** | 57.14% | 7.14% | -50.00% | — |
| **Rejection F1** | 0.3862 | 0.3741 | -0.0121 | — |
| **Avg Latency** | 2.95s | 2.97s | — | Viable (<5s) |
