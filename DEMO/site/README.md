# Drift-Sense++ Interactive Demonstration Interface

This directory contains the self-contained static presentation and demonstration dashboard deployed via **GitHub Pages** for the **Applied Materials Drift-Sense SEM Localization Phase 2 Challenge**.

## 🌐 Live Demonstration
👉 **[https://aashishniranjanb.github.io/Drift-Sense-SEM-Localization/](https://aashishniranjanb.github.io/Drift-Sense-SEM-Localization/)**

---

## 🔬 Purpose & Separation

- **Presentation / Demonstration:** This website is a static interactive visualization interface designed for judges, researchers, and technical reviewers to explore SEM imagery, candidate pool distributions, periodic replica disambiguation, and subpixel pose fitting.
- **Authoritative Competition Algorithm:** The actual Phase 2 scoring pipeline runs under [`FINAL_SUBMISSION/register.py`](../FINAL_SUBMISSION/register.py) adhering strictly to the offline reference machine contract. The hosted dashboard utilizes precomputed evaluation data from `data/samples.json` for fast browser rendering.

---

## 📁 Structure

```text
site/
├── index.html                 # Self-contained research dashboard (HTML5, Canvas, CSS3)
├── assets/
│   └── images/                # High-resolution SEM pairs, architecture diagrams, and charts
├── data/
│   └── samples.json           # Precomputed candidate pools, metrics, and ground-truth coordinates
└── README.md                  # Documentation and deployment notes
```

---

## 🚀 Deployment

Automated deployment is handled via GitHub Actions in [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml). Any push to the `main` branch automatically deploys the `site/` folder to GitHub Pages.

---

### ⚖️ Disclaimer
*Independent competition project / demonstration interface. Developed for the Applied Materials Drift-Sense SEM Localization Phase 2 Challenge. Not an official Applied Materials product or endorsement.*
