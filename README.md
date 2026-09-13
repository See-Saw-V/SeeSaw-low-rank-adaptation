<p align="center">
  <img src="figures/logo_final.png" alt="Seesaw logo" width="480"/>
</p>

<p align="center">
  <em>Official code release for <strong>"Seesaw: Budget-Preserving Rank Reallocation for Low-Rank Adaptation"</strong> (under review at ICLR 2027)</em>
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-under%20review-lightgrey">
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

Seesaw redistributes a fixed LoRA rank budget across Transformer
layers during training, using a lightweight gradient-derived
importance signal — without ever changing the total rank sum. It
separates *how much* adaptation capacity a model gets from *where*
that capacity is placed.

## Overview

Conventional LoRA assigns every layer the same rank, assuming uniform
adaptation demand. Seesaw instead:

- Tracks an EMA-smoothed importance score per module from gradient magnitude
- Identifies the highest-scoring receiver and lowest-scoring donor module
- Transfers a single rank unit between them when their gap clears a threshold τ
- Preserves the global rank budget exactly at every step (Σ ranks = R, always — proof in Appendix A.2)

Two instantiations are provided:
- **Seesaw-v1** — fixed-formula version, query-projection only, evaluated on BERT-base
- **Seesaw-v2** — generalized version with z-score normalization, Q/K/V/O projections, evaluated on DeBERTa-v1/v3

## Repository structure
