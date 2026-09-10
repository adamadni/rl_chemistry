# rl_chemistry

Reproduction of **Korshunova et al. 2022** (*Commun Chem* **5**:129) — a sparse-reward
reinforcement-learning molecular generator — retargeted to **ABL1 (CHEMBL1862)**, with
modifications to the policy-optimisation scheme that the original method needed in
order to converge here.

A character-level SMILES language model is pretrained on ChEMBL, then fine-tuned by
policy-gradient RL against a QSAR reward model. The objective is **novel, chemically
diverse scaffolds** predicted active against ABL1 — without the mode collapse that
this class of method is prone to.

---

## Headline result

| high-reward set (P(active) ≥ 0.5, n=5000) | baseline (v3) | final (v5d) |
|---|---|---|
| distinct chemotypes (generic frameworks) | 19 | **333** |
| share held by the single largest chemotype | 77% | **10%** |
| predicted-active rate | 5.7% | **14.9%** |
| applicability-domain coverage | 94.4% | **99.8%** |
| validity / uniqueness | 99.6% / 96.0% | 99.8% / 93.5% |

The final model expands chemotype diversity **17.5×** while simultaneously *increasing*
predicted potency and staying further inside the reward model's domain of validity.

Structure-based check: five generated candidates docked at the **known-actives mean**
(−11.17 kcal/mol vs 40 known ABL1 actives at −11.31 and 40 property-matched inactives
at −9.85), and one candidate reproduces the complete imatinib/nilotinib type-II binding
signature. No candidate outscores imatinib.

---

## Pipeline

```
ChEMBL 37 (2.9M compounds)
  └─ preprocess_chembl.py ──▶ 2,599,495 clean SMILES, 42-token char vocab
       └─ pretrain_generator.py ──▶ 3-layer LSTM prior (5.54M params)
                                    97.9% valid / 100% unique / 95.2% novel

ChEMBL ABL1 bioactivity (6,314 records)
  └─ pull_abl1.py ──▶ 3,097 compounds, 1,265 active (max-agg pChEMBL ≥ 8)
       └─ qsar_model.py ──▶ ECFP4 + RandomForest reward model
                            scaffold-split ROC-AUC 0.900 | 5-fold CV 0.948
                  ▼
  reward_model.py   ── reward = P(active) − λ·σ_trees(P(active))
                  ▼
  train_rl.py       ── PPO-clipped updates + adaptive KL controller
                       + threshold shaping                        [v3 baseline]
                       + diversity filter        (diversity_filter.py)   [F]
                       + experience replay       (replay_buffer.py)      [R]
                       + transfer learning       (transfer_phase)        [T]
                  ▼
  eval_policy.py / analyze_scaffolds.py  ── validity, uniqueness, novelty,
                                            frameworks, activity, applicability domain
                  ▼
  select_candidates.py ──▶ candidates + size-matched measured negatives
                  ▼
  prep_receptors.py → run_docking.py → interaction_fingerprint.py
                                            structure-based validation
```

## The three policy modifications

The paper's three stabilisers, implemented and ablated as a full 2³ factorial.

**F — diversity filter** (`src/diversity_filter.py`). Per-scaffold occupancy memory;
reward on an over-mined chemotype decays linearly to zero. Keys on the **Murcko generic
framework**, not the Bemis-Murcko scaffold: v3's top three "distinct" scaffolds were
phenyl / 2-pyridyl / 3-pyridyl on one core, so a Murcko-keyed filter is evadable by
moving a single ring nitrogen.

**R — experience replay** (`src/replay_buffer.py`). Scaffold-capped memory of high-reward
molecules, entering as an auxiliary per-token likelihood term — deliberately *not*
through the PPO ratio, because a molecule stored thousands of steps ago has a stale
`logP_old` that either saturates the clip or, if recomputed, silently reverts to
uncapped REINFORCE.

**T — transfer learning** (`transfer_phase` in `src/train_rl.py`). Periodic supervised
MLE on a scaffold-diverse slice of the buffer, with a **KL-to-prior tripwire** checked
after every epoch. Nothing inside an MLE phase otherwise bounds how far the policy moves.

### Ablation (n=5000 per cell)

| run | F | R | T | frameworks | top share | P≥0.5 | AD% |
|---|:-:|:-:|:-:|---|---|---|---|
| v3 | | | | 19 | 77% | 5.7% | 94.4 |
| v4a | ✓ | | | 231 | 4% | 11.4% | **45.2** |
| v4b2 | | ✓ | | 7 | 14% | 0.10% | 93.6 |
| v5a | | | ✓ | 19 | 47% | 5.7% | 97.0 |
| v4c | ✓ | ✓ | | 138 | 7% | 3.8% | 94.4 |
| v5b | ✓ | | ✓ | 192 | 7% | 5.9% | 96.6 |
| v5c | | ✓ | ✓ | 8 | 22% | 0.10% | 93.4 |
| **v5d** | ✓ | ✓ | ✓ | **333** | **10%** | **14.9%** | **99.8** |

Three findings:

1. **The diversity filter is the only component that produces diversity *and* learning
   on its own.** Every cell containing F reaches 138–333 frameworks; every cell without
   it either fails to learn or stays as narrow as the baseline.
2. **Experience replay alone does not work** — 0.10% predicted-active, the pretrained
   prior's own rate. It is a *consolidation* mechanism: with nothing generating
   high-reward molecules there is nothing to consolidate, and buffer admission requires
   reward ≥ 0.4, so it cannot bootstrap.
3. **The components are superadditive.** F alone drifts out of the reward model's domain
   (AD 45.2%); adding either memory mechanism repairs it (R → 94.4, T → 96.6, both →
   99.8). The filter *pushes* the policy off the mined chemotype; the buffer-backed
   mechanisms *anchor* it to chemistry already observed to score well.

The filter also **improved** optimisation stability rather than harming it: minimum
uniqueness 56.7% vs v3's 21.3%, and neither v4a nor v5d ever hit the `beta_kl` ceiling
(v3 hit it 14 times). Removing the single dominant reward hill removed the pressure
driving the baseline toward collapse.

---

## Structure-based validation

Independent of the QSAR model: gnina/smina docking against an ABL1 ensemble —
**1IEP** and **3CS9** (DFG-out, imatinib/nilotinib) and **2GQG** (DFG-in, dasatinib) —
benchmarked on 40 known actives vs 40 inactives matched to **0.00 heavy atoms** and
0.03 logP, since docking scores scale with molecular size.

- Best enrichment **AUC 0.797 [0.682, 0.901]**. Real but modest, and **weaker than the
  QSAR model it is checking** (0.900). Its value is independence, not accuracy.
- **AlphaFold receptors are unusable for this target.** The AF-P00519 model is DFG-in
  (Phe382–Glu286 9.63 Å vs 13.9 Å for both DFG-out crystals), so the type-II allosteric
  pocket imatinib and nilotinib require does not exist in it. Docking against it alone
  gives **AUC 0.400 — worse than random**. The fold itself is fine (kinase-domain pLDDT
  92.6, 2.07 Å CA-RMSD across N-lobe/hinge); the entire discrepancy is in the activation
  loop, exactly the part that decides type-II binding.
- **Pose validation reproduces known biology**: imatinib and nilotinib show the full
  type-II signature (hinge Met318 backbone H-bond, gatekeeper Thr315, αC Glu286, DFG
  Asp381) on DFG-out; dasatinib shows type-I hinge binding on DFG-in and *loses the hinge
  entirely* on DFG-out. Each drug behaves as its binding class predicts.
- **cand4** holds the complete type-II signature on both DFG-out structures and is the
  best-supported candidate. **cand2** is rejected by score *and* pose — the one
  unambiguous verdict.

---

## Limitations

Stated plainly, because several of these bound what the results can mean.

**The reward model is the ceiling.** Every "active" claim is a RandomForest prediction
from 3,097 ABL1 compounds. Nothing here is experimentally validated. The 14.9%
"hit rate" means 14.9% *by the RF's own judgement*.

**RF ensemble variance does not detect out-of-domain drift.** The reward penalises
`λ·σ_trees` as an applicability-domain guard, and v4a shows it failing: uncertainty
*fell* to 0.199 while AD coverage halved to 45.2%. Tree agreement stays high where the
forest never looked. Any future domain guard needs an explicit similarity term.

**The two validators disagree.** RF confidence and docking rank the five candidates in
close to inverted order — the RF's most confident molecule (P=1.000) docks at the 22nd
percentile. Neither is thereby wrong, but **no candidate is corroborated by both**.

**n=1 on every RL run.** All runs use seed 42. The large gaps (v5d vs v4b2) are safe;
gaps like v4c vs v5b are not, without 3–5 seeds.

**Docking caveats.** Rigid receptor (three crystal conformations for a notoriously
flexible kinase); one protonation microstate per ligand; crystal waters discarded, which
matters for kinases; and docking-score-to-affinity correlation is weak in general
(r ≈ 0.3–0.5). Results support *discrimination*, not predicted potency.

**Diversity and lead-optimisation are opposed objectives.** Sampling 40,000 molecules
from v5d, only **0.006%** contain cand5's core (3.5% for cand4). A model optimised hard
for scaffold spread is correspondingly unwilling to elaborate any one scaffold — so this
model is a poor lead-optimisation engine, by construction.

**The diversity metric is coarse.** Generic frameworks merge genuinely distinct
chemotypes that share a ring skeleton. It is also a poor *selection* criterion at small
n — an early candidate pick returned five "distinct framework" molecules that were one
chemotype with different cyclic substituents, since Murcko pulls substituent rings into
the scaffold.

**The objective is single-task.** Nothing optimises selectivity, synthesisability or
ADMET. Without selectivity pressure this proposes pan-kinase binders.

**Hyperparameters were not swept.** `--df-bucket-size 25` was a first guess that
happened to work; the potency/diversity frontier is probably navigable by that one knob.

**Validity is measured more strictly than OpenChem's.** Strict RDKit sanitisation, where
OpenChem's `sanitize_smiles()` tolerates valence errors — so numbers are not directly
comparable to the paper's, or to MOSES/GuacaMol.

**Relative binding free energy was set up but never run.** Inputs are prepared
(`src/prep_rbfe.py`, `src/run_rbfe.py`, 12 ligands embedded on the validated pose,
star map in `results/series/rbfe_cand4.json`), but no RBFE result exists — a full
campaign is 3–5 GPU-days and was judged out of scope. **These two scripts are setup
only and have never executed successfully.**

---

## Files

| Path | Role |
|---|---|
| `src/paths.py` | Project-root resolution — every other path derives from here |
| `src/preprocess_chembl.py` | ChEMBL cleaning, vocab, train/val split |
| `src/pretrain_generator.py` | Character-level LSTM prior |
| `src/pull_abl1.py` | ABL1 bioactivity, max-aggregated pChEMBL |
| `src/qsar_model.py` | ECFP4 + RF, scaffold-split validation |
| `src/reward_model.py` | `P(active) − λ·σ_trees` (deliberately unclipped) |
| **`src/train_rl.py`** | **The RL loop. Its module docstring documents every collapse and fix.** |
| `src/diversity_filter.py` | Component F — scaffold occupancy memory |
| `src/replay_buffer.py` | Component R — scaffold-capped replay |
| `src/eval_policy.py` | Policy vs prior metrics |
| `src/analyze_scaffolds.py` | Chemotype analysis of the high-reward tail |
| `src/select_candidates.py` | Candidates + size-matched measured negatives |
| `src/prep_receptors.py` | ABL1 ensemble prep, DFG-state analysis |
| `src/build_benchmark.py` | Property-matched actives/inactives benchmark |
| `src/run_docking.py`, `src/run_benchmark_docking.py` | smina / gnina docking |
| `src/analyze_docking.py` | Enrichment AUC with bootstrap CIs |
| `src/interaction_fingerprint.py` | Pose validation against ABL1 pharmacophore |
| `src/multiseed_table.py`, `src/final_table.py` | Result tables |
| `src/build_series.py`, `src/select_rbfe_set.py` | Congeneric series construction |
| `src/prep_rbfe.py`, `src/run_rbfe.py` | RBFE setup — **never successfully run** |
| `results/` | All metrics, per-run histories, docking results |
| `logs/` | Full training logs, every RL run |
| `ENGINEERING_LOG.md` | Detailed engineering log: every failure, diagnosis and fix, in the order they happened |

## Running it

All paths resolve from the repository root via `src/paths.py`, so a fresh clone works
unconfigured. Set `RL_CHEM_ROOT` only if data and checkpoints should live somewhere other
than the checkout (a mounted volume, a scratch disk).

```bash
git clone https://github.com/adamadni/rl_chemistry && cd rl_chemistry
pip install -r requirements-lock.txt          # or ./bootstrap.sh

# 1. build the corpus and the reward model (downloads ChEMBL; several hours)
python src/preprocess_chembl.py
python src/pretrain_generator.py
python src/pull_abl1.py
python src/qsar_model.py

# 2. the final model — 5000 steps, ~40 min on a 4090
python src/train_rl.py --steps 5000 --batch 128 --lr 1e-4 \
  --beta-kl 0.02 --target-kl 3.0 --ppo-epochs 4 --clip-eps 0.2 \
  --lambda-unc 1.0 --threshold-update-every 50 --threshold-decay 0.1 \
  --diversity-filter --replay --transfer-learning \
  --out checkpoints/rl_v5d

# 3. evaluate
python src/eval_policy.py --policy checkpoints/rl_v5d/policy_latest.pt --n 5000
python src/analyze_scaffolds.py --policy checkpoints/rl_v5d/policy_latest.pt --n 5000
```

Step 1 needs the ChEMBL 37 `chemreps` archive in `data/raw/`; steps 2–3 need a CUDA GPU.
The docking scripts additionally need `smina` or `gnina` binaries in `docking/`.

**Timings** (RTX 4090): full RL run ~40 min · full 8-cell ablation ~3 h · docking benchmark
(100 ligands × 3 receptors × 3 seeds) ~35 min.

**Not committed:** model weights, the ChEMBL corpus and the environment — size, not secrecy.
The repository holds code and metrics.

**Watch during a run:** `unique_pct` and `scaffold_ratio` (collapse), `beta_kl` (spikes
when `kl` rises — the controller catching a runaway), `dfmult` (filter bite),
`pct_above_tau` (learning signal). Always pass `--threshold-decay` when the filter or
replay is on: with monotonic tau the bar can strand above everything reachable and the
run silently stops learning while still looking healthy at 99% uniqueness.
