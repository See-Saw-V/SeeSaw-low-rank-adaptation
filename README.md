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


## Scope of released code

- **Table 3** (DeBERTa-v3-base vs. established PEFT baselines): SST-2, CoLA,
  STS-B, and MRPC — **complete**, each with its own script in `experiments/`.
- **Importance Normalization Ablation** (Appendix A.1, A.4): raw, mean-divided,
  and z-score scoring, at two controller checkpoint frequencies — **complete**,
  see `ablations/normalization/`.
- **Table 1** (BERT-base, 6 GLUE tasks): reference implementation for
  AG News (`experiments/table1_ag_news_bert_base.py`). The remaining
  five tasks use the identical Seesaw-v1 training loop with the
  corresponding GLUE subset substituted.
- **Table 2** (DeBERTa-v1 efficiency profiling): the Seesaw row is
  included; baselines (Full FT, LoRA, LoRA-FA, (IA)³) use standard
  reference implementations at matched configuration.
- The Fixed Static, Inverted Dynamic (Section 6), and Optimizer-Reset
  Policy ablations are direct modifications of the training loop in
  `experiments and are not included as
  separate scripts in this release.

## Installation

```bash
git clone https://github.com/<username>/seesaw-lora.git
cd seesaw-lora
pip install -r requirements.txt
```

## Usage

```bash
python experiments/table3_cola_deberta_v3.py
```

Hyperparameters are defined at the top of each script and differ
across tasks/backbones — see Appendix B (Per-Task Hyperparameters) in
the paper for the full breakdown.

## Citation

```bibtex
@inproceedings{anonymous2027seesaw,
  title     = {Seesaw: Budget-Preserving Rank Reallocation for Low-Rank Adaptation},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2027},
  note      = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This repository is submitted as part of an anonymous double-blind
review process for ICLR 2027. Author names and identifying details
will be added upon acceptance.
