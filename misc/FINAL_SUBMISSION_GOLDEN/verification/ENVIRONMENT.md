# Drift-Sense++ Execution Environment Specification

## System Requirements
- **Architecture:** x86_64 CPU (minimum 4 physical cores recommended).
- **RAM:** Minimum 4 GB, recommended 8 GB.
- **Operating System:** Linux (Ubuntu 20.04+), macOS (12.0+), or Windows (10/11).
- **Accelerator:** None. **GPU is NOT required or used.**
- **Network:** **No internet connection needed.** All model weights and stage caches are bundled.

## Pinned Runtime Environment
- **Python Version:** 3.11.x (compatible with Python 3.9 through 3.14).
- **Core Dependencies:**
  ```text
  numpy==2.4.0
  scipy==1.16.3
  scikit-learn==1.8.0
  opencv-python-headless==4.13.0.92
  pandas==2.3.3
  joblib==1.5.3
  threadpoolctl==3.6.0
  ```

## Hardware Resource Invariants
- Single-thread BLAS/LAPACK threads default to 1 (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
- OpenCV threading fixed to 1 (`cv2.setNumThreads(1)`).
- Deterministic behavior guaranteed via fixed PRNG seed (`42`).
