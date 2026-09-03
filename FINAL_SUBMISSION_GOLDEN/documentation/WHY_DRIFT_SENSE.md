# Why Drift-Sense++? Foundational Design Decisions

In nanoscale semiconductor localization, naive pattern matching breaks down because physical chip layouts violate standard computer vision assumptions. 

This document answers the five foundational architectural questions that shaped Drift-Sense++.

---

## 1. Why Not Raw NCC?
**The Problem:** Normal cross-correlation (NCC) operates on the assumption that the true physical location produces a distinct, global maximum in the correlation plane.
**The SEM Reality:** In high-density memory (DRAM) and logic (FinFET) arrays, adjacent unit cells are separated by just tens of nanometers and share identical geometries. In degraded SEM images with shot noise and edge shadowing, neighboring replicas produce correlation values within $\Delta\text{NCC} < 0.005$ of the true site. Raw NCC greedily traps on whichever replica happened to receive a stochastic noise spike.
**Our Solution:** Drift-Sense++ never treats raw NCC as a final decision. NCC is used strictly as a candidate proposal filter.

---

## 2. Why Candidate Generation?
**The Problem:** Searching the entire continuous 4-D parameter space $(x, y, \theta, \text{scale})$ simultaneously using non-linear optimization (gradient descent or Nelder-Mead) is susceptible to local minima in periodic arrays.
**The SEM Reality:** An optimizer initialized near a false replica will converge to the wrong physical cell with high confidence.
**Our Solution:** Drift-Sense++ decouples search into two distinct phases:
1. Frequency-domain coarse FFT sweep proposes a discrete pool of the top **200 spatial candidate peaks**.
2. A learned multi-evidence discriminator evaluates candidates in parallel, guaranteeing global capture range without local minima trapping.

---

## 3. Why Replica-Family Reasoning?
**The Problem:** Evaluating candidates independently ignores the fundamental physics of semiconductor manufacturing: features repeat along regular Bravais lattice vectors.
**The SEM Reality:** Correlation peaks from periodic structures do not occur at random positions—they form periodic grids with fixed pitch $(\Delta x_p, \Delta y_p)$.
**Our Solution:** Drift-Sense++ clusters candidate peaks into **periodic replica families**. By measuring family population density and lattice regularity, the system distinguishes genuine isolated features from background array noise and prevents replica hopping.

---

## 4. Why Extended Context?
**The Problem:** A localized template patch (e.g. $64\times 64$ pixels) centered on a transistor gate contains only local parallel lines—completely indistinguishable from adjacent gates.
**The SEM Reality:** While local features are ambiguous, global chip layouts are non-periodic at larger scales (e.g. array boundaries, tap cells, guard rings, power rails).
**Our Solution:** Drift-Sense++ extracts an **extended $128\times 128$ contextual surround** around each candidate. Even if the core patch is ambiguous, the extended boundary context provides the symmetry-breaking signal needed to isolate the unique physical coordinates.

---

## 5. Why Explicit Geometry Instead of Deep Learning?
**The Problem:** Deep neural networks (CNNs, Vision Transformers) are trained for semantic object classification ("transistor" vs. "background").
**The SEM Reality:**
- SEM metrology is a **geometric alignment** problem requiring subpixel precision ($\le 0.2\text{ px}$). Convolutional downsampling destroys spatial phase.
- Deep nets hallucinate feature positions under severe SEM noise and charging halos.
- In semiconductor yield analysis, silent false accepts of absent defects are catastrophic; deep neural networks provide no verifiable, inspectable failure bounds.
**Our Solution:** Drift-Sense++ uses **explicit geometric physics**: Fourier-domain phase correlation, steerable Sobel gradient orientation matching, and 2-D analytical paraboloid surface fitting. This guarantees:
- Subpixel accuracy without deconvolutional artifacts.
- Deterministic, mathematically provable behavior.
- Ultra-low latency (0.07 s/pair) on standard CPU with zero GPU requirements.
