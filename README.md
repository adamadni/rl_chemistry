# rl_chemistry

Reproduction of **Korshunova et al. 2022** (*Commun Chem* 5:129) — a sparse-reward
RL molecular generator — targeting **ABL1 (CHEMBL1862)**.

A pretrained SMILES language model is fine-tuned with policy-gradient RL against a
QSAR reward model, with the goal of proposing **novel, chemically diverse scaffolds**
predicted active against ABL1 — without collapsing onto a single answer.

> **Compute lives on a remote RunPod RTX 4090.** This directory is a local mirror of
> the code and results for review. Data, model weights and the venv stay on the pod's
> persistent network volume (`/workspace/rl_chemistry`). See `CLAUDE.md` for pod access.

---

## Pipeline

```
ChEMBL 37 (2.9M compounds)
   └─ preprocess_chembl.py ──▶ 2,599,495 clean SMILES, 42-token char vocab
         └─ pretrain_generator.py ──▶ 3-layer LSTM prior (5.54M params)
                                       97.9% valid / 100% unique / 95.2% novel

ChEMBL ABL1 bioactivity (6,314 records)
   └─ pull_abl1.py ──▶ 3,097 compounds, 1,265 active (max-agg, pChEMBL ≥ 8)
         └─ qsar_model.py ──▶ ECFP4 + RandomForest reward model
                               scaffold-split ROC-AUC 0.900 | 5-fold CV 0.948

                    ▼
   reward_model.py  ──  P(active) − λ·σ_trees(P(active))
                    ▼
   train_rl.py      ──  PPO-clipped policy gradient + adaptive KL + threshold shaping
                    ▼
   eval_policy.py   ──  policy vs. prior: validity / uniqueness / novelty /
                        scaffold count / predicted-active / applicability domain
```

## Files

| File | Role |
|---|---|
| `src/preprocess_chembl.py` | ChEMBL corpus cleaning, vocab, train/val split |
| `src/pretrain_generator.py` | Character-level LSTM language model (the RL prior) |
| `src/pull_abl1.py` | Pulls ABL1 bioactivity, **max**-aggregates pChEMBL per compound |
| `src/qsar_model.py` | ECFP4 + RF classifier, Bemis-Murcko scaffold-split validation |
| `src/reward_model.py` | Reward = P(active) − λ·ensemble-uncertainty |
| **`src/train_rl.py`** | **The RL loop. Read its module docstring — it documents every failure and fix.** |
| `src/eval_policy.py` | Compares a finetuned policy against the pretrained prior |
| `src/smiles_utils.py` | Strict RDKit validity/canonicalization (see caveat below) |
| `src/diag_drugs.py` | Diagnostic: why known drugs scored as they did |
| `results/` | Metrics: `qsar_metrics.json`, `rl_baseline_v3/{history,eval}.json` |
| `logs/` | Full training logs for all three RL runs (v1, v2 collapsed; v3 did not) |

## Reproducing (on the pod)

```bash
cd /workspace/rl_chemistry && source venv/bin/activate

# RL training — ~39 min for 5000 steps on the 4090
nohup python src/train_rl.py --steps 5000 --batch 128 --lr 1e-4 \
  --beta-kl 0.02 --target-kl 3.0 --ppo-epochs 4 --clip-eps 0.2 \
  --lambda-unc 1.0 --threshold-update-every 50 \
  --out checkpoints/rl_baseline_v3 > logs/rl_baseline_v3.log 2>&1 &

# Evaluation
python src/eval_policy.py --policy checkpoints/rl_baseline_v3/policy_latest.pt --n 5000
```

**Watch during a run:** `unique_pct` and `scaffold_ratio` (collapse signal),
`beta_kl` (should spike when `kl` rises — that's the controller catching a runaway),
`pct_above_tau` and `p_act` (learning signal).

## Deviations from the paper

The paper's method as written collapsed to a single molecule in our hands, twice.
Five changes were needed for stable convergence — all documented in detail in the
`src/train_rl.py` module docstring, summarized in `CLAUDE.md`.

## Known caveats

- **Validity metric differs from OpenChem's.** We use strict RDKit
  (`smiles_utils.is_valid`, sanitization on). OpenChem's `sanitize_smiles()` accepts
  valence errors, so its validity numbers are not directly comparable to ours or to
  MOSES/GuacaMol.
- **The reward model is a surrogate.** All "active" claims are RF predictions, not
  measurements. The RF's own uncertainty is penalized in the reward, and
  applicability-domain coverage is tracked, but nothing here is experimental validation.
- **The high-reward tail is still narrow** — see `CLAUDE.md`. Global diversity is
  healthy; diversity *among top-scoring molecules* is the open problem.
