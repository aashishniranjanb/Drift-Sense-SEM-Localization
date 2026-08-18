# Component 2 — GitHub Repository Mandatories & Documentation

This directory contains all mandatory files for **Component 2 — GitHub Repository Submission** as specified by the Hackathon guidelines:

1. **`README.md`**: Complete, self-contained setup and execution instructions.
2. **`dataset_generator.py`**: Standalone dataset generator accepting `--architecture`, `--num-pairs`, and `--output-dir`.
3. **`inference.py`**: Standalone competition localization inference script accepting `--reference` and `--search` and returning `(x, y)`.
4. **`models/pace_best.pt`**: Deep Learning model weights loaded automatically by `inference.py`.
5. **`train_pace_ranker.py`**: Complete deep learning training script reproducing PACE Group List-Ranking model weights.
6. **`requirements.txt`**: Complete pip freeze dependencies list.
7. **`REFERENCES_CITATIONS.md`**: Literature and patent references justifying noise models, FFT registration, and process-aware overlap matching.

---

## 1. Quick Start Execution Test

### Running Localization Inference (Applied Materials Scoring Test)
```bash
python inference.py --reference data/benchmark_120/reference/0000.png --search data/benchmark_120/search/0000.png --verbose
```
**Output**:
```json
{
  "x": 305.09,
  "y": 620.88,
  "ncc_score": 0.9471,
  "delta_s": 0.0898,
  "psr": 2.97,
  "consensus_D_px": 0.31,
  "is_confident": true,
  "path": "FAST_TRUSTED_FFT",
  "pace_activated": false,
  "latency_ms": 59.61,
  "status": "OK"
}
(305.09, 620.88)
```

### Running Dataset Generator
```bash
python dataset_generator.py --architecture DRAM --num-pairs 10 --output-dir data/sample_generated
```

### Running Master Dual-Channel Ablation Benchmark
```bash
python benchmark_car_ablation.py
```
