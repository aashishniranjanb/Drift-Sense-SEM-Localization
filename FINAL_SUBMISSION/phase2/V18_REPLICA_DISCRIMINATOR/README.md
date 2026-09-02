# Phase V18: Replica Discriminator 2.0

## 1. Mission Overview
Convert the empirical discoveries of Phase V17 into a robust candidate identity discriminator.
Target: Resolve the **17 ranking failure cases** where Ground Truth was present in the candidate pool but lost to a peripheral periodic replica.

**Primary Goal:** Increase conditional Top-1 accuracy on candidate pool from **60% (V14/V16 control)** to **$\ge 75\%$ (stretch $\ge 80\%$)** without degrading Set A or non-periodic nominal cases.

---

## 2. Controlled Experiment Ladder
- **V18-A**: Periodicity-Adaptive Center Prior alone ($NCC + w_{\text{fam}} \cdot \text{CenterPrior}$).
- **V18-B**: Multi-Scale Context + Adaptive Center Prior.
- **V18-C**: Phase Residual Gate + Adaptive Center Prior.
- **V18-D**: Full Handcrafted Normalized Composite Evidence ($NCC + \text{PSR} + \text{Phase} + \text{Context} + \text{Center} + \text{Fingerprint}$).
- **V18-E**: Machine-Learned Pairwise Identity Classifier (Logistic Regression / Gradient Boosting).
