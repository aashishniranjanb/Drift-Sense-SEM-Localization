# V22 Championship Fusion Results

## V22-A Control (V21 Baseline)
- Total Score: 50.00
- Localization: 4.91
- Pose: 19.55
- Rejection: 5.32
- Calibration: 5.21
- Efficiency: 5.0
- Docs: 10.0
- Rejection F1: 0.354
- Calibration AUC: 0.521
- PRESENT recall: 0.457
- Set B <= 5px: 0.071

## Best V22 Model (V22-F Calibrated HistGradientBoosting)
- Threshold T*: 0.10
- Total Score: 50.36
- Localization: 4.14
- Pose: 20.0
- Rejection: 5.29
- Calibration: 5.92
- Efficiency: 5.0
- Docs: 10.0
- Rejection F1: 0.352
- Calibration AUC: 0.592
- PRESENT recall: 0.285
- Set B <= 5px: 0.071

## Ablation (Val AUC)
- V22-B Logistic Regression: 0.8516
- V22-C Global Validator: 0.8438
- V22-D Nonlinear: 0.8906
- V22-E Hard Negative: 0.8906
- V22-F Calibrated: 0.9219
