# TARGET.md: Drift-Sense++ HCR (Hard-Negative Candidate Re-ranking) Roadmap & Targets

## Executive Vision & Technical Direction

**Drift-Sense++ HCR** is a hybrid framework combining **Global Candidate Retrieval**, **Learned Hard-Negative Structural Discrimination**, and **Classical Metrology-Grade Subpixel Registration** for nanometre-accurate SEM navigation error recovery on wafer inspection tools.

---

## 1. Division of Responsibilities

```
┌──────────────────────────────────────────────────────────────────────────┐
│ AI / Metric Learning (Siamese Discriminator):                            │
│ "WHICH candidate is the true physical site vs. periodic replica?"        │
├──────────────────────────────────────────────────────────────────────────┤
│ Classical Registration (Phase Correlation + Paraboloid Fit):             │
│ "WHERE EXACTLY is the site centroid in subpixel space?"                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Target Engineering Milestones & Metrics

| Tier | Metric Goal ($\le 5$ px) | Metric Goal ($\le 1$ px) | Equivalent Physical Error | Target Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Minimum Credible** | $> 70\%$ | $> 65\%$ | $< 50$ nm | $< 350$ ms |
| **Tier 2: Strong Competition** | $> 80\%$ | $> 75\%$ | $< 10$ nm | $< 250$ ms |
| **Tier 3: Metrology-Grade** | $\mathbf{\ge 90\%}$ | $\mathbf{\ge 85\%}$ | **$< 3.6$ nm ($< 0.36$ px)** | $< 200$ ms |

### Physical Error Conversion (at $\approx 10$ nm/pixel search resolution):
- $5.0$ px error $\approx 50$ nm
- $1.0$ px error $\approx 10$ nm
- $0.36$ px error $\approx 3.6$ nm *(Targeting published SEM in-die overlay repeatability)*
- $0.10$ px error $\approx 1.0$ nm

### Primary Optimization Objective:
$$\mathcal{J} = P(\text{Correct Site Selected}) \times P(e_{\text{subpixel}} < 1\text{ px} \mid \text{Correct Site}) - \lambda \cdot \text{Latency}$$

---

## 3. End-to-End Architecture Pipeline

```mermaid
flowchart TD
    subgraph Inputs [Input Image Pair]
        Ref[100x Reference Image @ 1 nm/px]
        Search[10x Search Image @ 10 nm/px]
    end

    subgraph Stage0 [Stage 0: Physical Normalization]
        Norm[Downsample Ref ~10x to 100x100 Template + Contrast Normalization]
    end

    subgraph Stage1 [Stage 1: Transform Estimation & Fast Retrieval]
        FM[Lightweight Fourier-Mellin / Scale-Rotation Estimator]
        FFT[Multi-Channel FFT-NCC: Intensity + Scharr Gradient]
        TopK[Generate Top-20 / Top-30 Spatial Candidates via NMS]
    end

    subgraph Stage2 [Stage 2: Learned Structural Re-Ranking]
        Siam[Multi-Scale Siamese CNN Encoder: 64x64 Local + 128x128 Context]
        Embed[Compute Structural Embeddings: z_R vs {z_C1 ... z_C20}]
        Rank[Hard-Negative Discriminated Top-3 Candidates]
    end

    subgraph Stage3 [Stage 3: Classical Metrology Verification]
        Verif[Local Phase Correlation + Gradient Coherence + Orientation Alignment]
    end

    subgraph Stage4 [Stage 4: Periodicity Gate & Conditional Prior]
        Gate{Ambiguity Check: Delta-S < tau AND Periodic Lattice Confirmed?}
        CenterRule[Apply Center-Nearest Tie-Breaker]
        BestScore[Select Top Structural Candidate]
    end

    subgraph Stage5 [Stage 5: Subpixel Refinement]
        Subpix[5x5 Quadratic Paraboloid Surface Fit]
        Output["Final Output: (x, y)"]
    end

    Inputs --> Stage0
    Stage0 --> Stage1
    FM --> FFT
    FFT --> TopK
    TopK --> Stage2
    Siam --> Embed
    Embed --> Rank
    Rank --> Stage3
    Stage3 --> Stage4
    Gate -- Yes --> CenterRule
    Gate -- No --> BestScore
    CenterRule --> Stage5
    BestScore --> Stage5
    Subpix --> Output
```

---

## 4. Key Architectural Innovations

### 4.1 Learned Hard-Negative Structural Re-Ranker
- **Problem**: Whole-template FFT-NCC produces identical correlation peaks ($\Delta S < 0.005$) across repetitive DRAM/FinFET lines.
- **Solution**: A lightweight Siamese CNN with multi-scale context ($64 \times 64$ local + $128 \times 128$ context).
- **Training Strategy**: Hard-negative mining.
  - Positive pair: $(R, P_{\text{true}})$.
  - Hard negative: $(R, N_{\text{periodic}})$ — adjacent periodic array replicas where FFT-NCC score is high but physical location is wrong.
  - Triplet loss: $\mathcal{L} = \max(0, \|\mathbf{z}_R - \mathbf{z}_P\|_2 - \|\mathbf{z}_R - \mathbf{z}_N\|_2 + m)$.

### 4.2 Multi-Scale Context Windows
- $64 \times 64$ px: Captures fine contact pads, line ends, gate crossings.
- $128 \times 128$ px: Captures surrounding macro-boundaries, power straps, and array edge terminations.
- Embedding concatenation: $\mathbf{z} = [\mathbf{z}_{64}, \mathbf{z}_{128}] \in \mathbb{R}^{128}$.

### 4.3 Normalized Candidate Score Fusion
$$\mathcal{Z}_k(S) = \frac{S_k - \mu_k}{\sigma_k + \epsilon}$$
$$S_{\text{fusion}} = w_R \mathcal{Z}(S_{\text{NCC}}) + w_G \mathcal{Z}(S_{\text{Grad}}) + w_N \mathcal{Z}(S_{\text{Neural}}) + w_P \mathcal{Z}(S_{\text{Phase}})$$
Weights $(w_R, w_G, w_N, w_P)$ calibrated via logistic ranking regression on validation data.

### 4.4 Periodicity as a Decision Gate, Not a Location Signal
- **Rule**: Structural evidence strictly outranks spatial priors.
- Center-proximity selection is applied **only** when top candidates are structurally tied ($|S_1 - S_2| < \tau$) and confirmed as periodic lattice replicas.

---

## 5. Implementation & Experimentation Roadmap

1. **Dataset Split & Zero-Leakage Calibration**:
   - `data/train_hard_negatives` (synthetic dataset specifically mined for periodic false matches).
   - `data/val_calibration` (held-out tuning set for threshold/weight calibration).
   - `data/test_benchmark` (strictly unseen held-out test benchmark).
2. **Lightweight Siamese Model**:
   - Small CNN encoder ($< 500\text{k}$ parameters) with PyTorch / ONNX export.
   - Batch inference on GPU/CPU ($< 15$ ms for 20 candidate patches).
3. **Hard-Negative Mining Loop**:
   - Generate synthetic sample $\rightarrow$ Run FFT-NCC $\rightarrow$ Mine top false-positive peaks $\rightarrow$ Update training buffer $\rightarrow$ Train encoder.
4. **Classical Metrology Engine**:
   - Local phase correlation + 2D paraboloid subpixel refinement ensuring sub-0.36 pixel precision for correct sites.
5. **Standalone Competition CLI Interface**:
   - `python inference.py --reference <ref.png> --search <search.png>` $\rightarrow$ outputs `(x.xx, y.yy)`.
