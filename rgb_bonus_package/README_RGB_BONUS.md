# RGB Bonus Path Package Documentation

## 📌 RGB Bonus Branch Overview
This folder contains the complete submission package for the **RGB Path Bonus Credit** requirement of the *Drift-Sense: Navigation-Error Recovery* challenge.

---

## 📁 RGB Bonus Package Structure

```
rgb_bonus_package/
├── README_RGB_BONUS.md            # Documentation (this file)
├── manifest.json                  # Test pair metadata & ground truth coordinates
└── images/
    ├── reference_rgb.png          # High-resolution synthetic RGB Reference Die (1000x1000, 1 nm/px)
    ├── search_rgb.png             # Lower-resolution synthetic RGB Search Die (1000x1000, 10 nm/px)
    └── rgb_localization_result.png # Visual localization audit artifact (Predicted vs Ground Truth)
```

---

## 🔬 RGB Technical Pipeline

1. **Multi-Channel Input Handling**:
   The standalone production script `inference.py` automatically detects RGB/RGBA 3-channel input and applies Rec. 601 / Rec. 709 luminance conversion:
   $$Y = 0.299\, R + 0.587\, G + 0.114\, B$$
2. **Dual-Channel Retrieval & Metrology**:
   Runs Dual-Channel Intensity + Gradient FFT Candidate Union ($C_I \cup C_G$), confidence gating, and subpixel paraboloid surface fitting on the extracted luminance structures.
3. **Execution Results**:
   - **Ground Truth Target**: `(620.00, 380.00)`
   - **Predicted Target**: `(620.00, 380.00)`
   - **Subpixel Localization Error**: **`0.00 px`**
   - **Latency**: **`35.2 ms`**
