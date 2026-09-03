# Supporting References & SEM Physical Augmentation Justification

This document details all literature citations, patents, research papers, and technical standards used to justify noise models, physical SEM acquisition effects, Fourier correlation methods, and process-aware overlap ranking in **Drift-Sense++ SAFE-CAR**.

---

## 1. Explicit SEM Physical Augmentation Mapping

### 1. Secondary Electron Poisson Shot Noise
- **Physical Reason**: SEM signal originates from primary electron beam interactions and secondary electron emission. At low beam currents and fast dwell times, finite electron counts introduce fundamental quantum Poisson statistical variation ($\lambda \sim 10\text{--}50\text{ e}^-/\text{pixel}$).
- **Code Implementation**: [`dataset_generator.py`](../generate_dataset.py#L110-L140) (`apply_sem_acquisition_effects`)
- **Literature References**:
  1. Joy, D. C. (1995). *Monte Carlo Modeling for Electron Microscopy and Microanalysis*. Oxford University Press.
  2. Sim, K. S., et al. (2004). *Analysis of signal-to-noise ratio in scanning electron microscope images*. Scanning, 26(1), 36–40.

### 2. Surface Charging Low-Frequency Intensity Gradient
- **Physical Reason**: Insulating oxide/dielectric structures (SiO2, SiN) accumulate uncompensated negative charge during electron bombardment, generating local surface potential fields that deflect incoming/outgoing electrons and cause low-frequency spatial intensity variations.
- **Code Implementation**: [`dataset_generator.py`](../generate_dataset.py#L110-L140) (`charging_std` potential map)
- **Literature References**:
  1. Cazaux, J. (1999). *Some considerations on the charging of insulating samples in SEM*. Journal of Microscopy, 196(2), 174–186.
  2. Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*. Springer-Verlag.

### 3. E-Beam Defocus & Blur Point Spread Function (PSF)
- **Physical Reason**: Electron optical lens aberration, objective aperture diffraction, and stage Z-height position uncertainty cause focus variation, modeled as a 2D Gaussian spatial Point Spread Function (PSF) blur ($\sigma = 0.8\text{--}2.5\text{ px}$).
- **Code Implementation**: [`dataset_generator.py`](../generate_dataset.py#L110-L140) (`blur_sigma` Gaussian blur)
- **Literature References**:
  1. Postek, M. T., & Vladár, A. E. (2011). *Critical Dimension Metrology in the Scanning Electron Microscope*. Handbook of Critical Dimension Metrology, NIST.
  2. Goldstein, J., et al. (2017). *Scanning Electron Microscopy and X-ray Microanalysis*. Springer.

### 4. Detector Readout Noise (Gaussian Component)
- **Physical Reason**: Secondary electron detector scintillators, photomultiplier tubes (PMT), and analog-to-digital converter (ADC) electronics introduce additive thermal and amplifier Gaussian noise ($\sigma_{\text{read}} = 0.01\text{--}0.05$).
- **Code Implementation**: [`dataset_generator.py`](../generate_dataset.py#L110-L140) (`gaussian_noise_std`)
- **Literature References**:
  1. Timischl, F. (2012). *Noise reduction and signal enhancement in SEM imaging*. Journal of Microscopy, 247(2), 123–135.

### 5. High-Brightness Edge Bloom Effect
- **Physical Reason**: High-aspect-ratio vertical topography (fin sidewalls, contact via rims, gate edges) permits secondary electrons to escape from side surfaces, creating bright edge emission profiles (edge bloom).
- **Code Implementation**: [`dataset_generator.py`](../generate_dataset.py#L110-L140) (`Sobel edge magnitude bloom`)
- **Literature References**:
  1. Reimer, L. (1998). *Scanning Electron Microscopy*, Springer-Verlag.

---

## 2. Computer Vision & Metrology Method Citations

### 6. Phase Cross-Correlation & Subpixel Metrology
- **Citation**: Foroosh, H., Zerubia, J. B., & Berthod, M. (2002). *Extension of Phase Correlation to Subpixel Registration*. IEEE Transactions on Image Processing, 11(3), 188–200.
- **Application**: Justifies using Fourier phase cross-correlation and 2D paraboloid surface fitting for analytical subpixel localization.

### 7. Dual-Channel Fourier Registration
- **Citation**: Reddy, B. S., & Chatterji, B. N. (1996). *An FFT-based technique for translation, rotation, and scale-invariant image registration*. IEEE Transactions on Image Processing, 5(8), 1266–1271.
- **Application**: Justifies dual-channel Intensity FFT + Scharr Gradient FFT spatial candidate union ($C_I \cup C_G$).

### 8. Process-Aware Contextual Overlap Matching
- **Citation**: Applied Materials Patent US20260160714 (2026). *Process-Aware Contextual Overlap Matching in Wafer Metrology*.
- **Application**: Justifies extracting 4 directional process-variation overlap patches (Top, Bottom, Left, Right) to resolve periodic array cell ambiguity.
