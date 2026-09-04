# Drift-Sense++ Official Run Commands

### 1. Judge Quick Verification (7 automated audit stages)
```bash
python FINAL_SUBMISSION/verification/run_all.py
```

### 2. Official Competition Scoring Entry Point
```bash
cd FINAL_SUBMISSION
pip install -r requirements.txt
python register.py --input <path_to_pairs.csv> --output <path_to_predictions.csv>
```

### 3. Component 2 Standalone Localizer Interface
```bash
cd FINAL_SUBMISSION
python inference.py --reference <reference.png> --search <search.png>
```

### 4. Synthetic Evaluation Pair Generator
```bash
cd FINAL_SUBMISSION
python generate_dataset.py --style dram --num_pairs 5 --output_dir ./eval_data --seed 42
```
