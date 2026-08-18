# Standalone Demo Package (`demo/`)

This directory contains standalone test image pairs and visual localization outputs for rapid pre-submission verification.

## Quick Execution Test

Run localization inference on the clean success demo pair:
```bash
python inference.py --reference demo/reference.png --search demo/search.png --verbose
```

### Expected Output:
```json
{
  "x": 305.09,
  "y": 620.88,
  "confidence_score": 0.7654,
  "mode": "CLASSICAL",
  "decision": "LOCALIZED",
  "uncertainty": "LOW",
  "status": "OK",
  "path": "FAST_TRUSTED_FFT",
  "latency_ms": 34.88
}
(305.09, 620.88)
```

## Generated Visual Outputs
- [`demo/output_success.png`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/demo/output_success.png): High-precision localization result (Error = 0.20 px).
- [`demo/output_ambiguous.png`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/demo/output_ambiguous.png): Periodic array shift failure analysis visualization.
