# V11 Main Track Decision Document

This document records the keeping, modifying, or rejecting decisions for each of the V11 candidate-recovery experiments.

---

## 1. Experiment 1: Local Maxima Candidate Extraction
*   **Hypothesis**: Finding distinct spatial local maxima in a local window avoids the NMS replica-wiping problem.
*   **Result**: Window size $w=4$ ($9 \times 9$ neighborhood) raised Top-100 recall from **49.29% to 57.14%** without increasing compute overhead.
*   **Decision**: **KEEP**

---

## 2. Experiment 2: Suppression Radius Sweep
*   **Hypothesis**: The standard suppression radius of 15 pixels is too large and suppresses adjacent periodic structures.
*   **Result**: Reducing suppression radius to $r = 10$ pixels raised Top-50 recall to **51.4%** and Top-100 recall to **59.3%**.
*   **Decision**: **KEEP**

---

## 3. Experiment 3: Joint Scale-Rotation Search
*   **Hypothesis**: A joint coarse scale-rotation grid search avoids sequential error propagation.
*   **Result**: Resolves rotation coupling, but `cv2.warpAffine` interpolation introduces sub-pixel smoothing blur, which reduces peak-to-noise ratio in correlation planes under shot noise.
*   **Decision**: **MODIFY** (Perform sharp independent scale search but retain Top-3 coarse scale hypotheses).

---

## 4. Experiment 4: Multi-Hypothesis Pose Retrieval
*   **Hypothesis**: Keeping multiple scale/rotation hypotheses and unioning their candidate pools prevents losing target peaks.
*   **Result**: Retaining 3 scale-rotation hypotheses raised Top-100 recall from **48.6% to 57.1%** and Top-50 recall to **52.9%**.
*   **Decision**: **KEEP**
