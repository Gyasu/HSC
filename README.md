# HuSC — Human Spatial Constraint

**HuSC** (Human Spatial Constraint) is a computational framework for quantifying intraspecies constraint on missense variants by integrating population-scale human genetic variation with 3D protein structures. HuSC scores model the expected frequency of missense variation under neutral evolution and compare it to observed variation, accounting for both variation in mutational processes and 3D structural context.

We use HuSC to fine-tune protein language models (PLMs), improving their ability to predict variant effects — particularly by reducing bias toward wild-type sequences in regions that tolerate variation.

> **Associated publication:** Bajracharya G. and Capra J.A. "Fine-tuning protein language models on human spatial constraint improves variant effect prediction by reducing wild-type sequence bias." *bioRxiv* (2025).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gyasu/HuSC.git
cd HSC
pip install -e .
```