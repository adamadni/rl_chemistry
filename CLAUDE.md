# rl_chemistry — status

Reproduction of Korshunova et al. 2022 (Commun Chem 5:129), sparse-reward RL
molecular generator. Target: **ABL1 (CHEMBL1862)**, chosen by the user.

All compute is remote — see memory `runpod-pod-access` for the SSH wrapper
(RunPod's proxy needs it; plain `ssh host "cmd"` fails). Pod IDs change on
recreate; ask the user for the current SSH string. Project root
`/workspace/rl_chemistry` on the persistent network volume (survives pod
termination). `bootstrap.sh` rebuilds the venv on a fresh pod.

## Done
- Environment: RTX 4090, torch 2.8.0+cu128, RDKit 2026.03.5, OpenChem
  importable (see memory `openchem-python312-gotchas`).
- ChEMBL 37 pulled + sha256-verified (2,897,819 cpds).
- Corpus preprocessed → 2,599,495 unique molecules (93.2% pass), 42-token
  char vocab, train 2,469,521 / val 129,974. `src/preprocess_chembl.py`
- **Generator pretrained** (`src/pretrain_generator.py`): 3-layer LSTM, 5.54M
  params, 10 epochs, val loss 0.4912, 97.9% valid / 100% unique / 95.2% novel.
  → `checkpoints/generator_best.pt`
- ABL1 data pulled: 6,314 activity records → 3,097 unique compounds,
  1,186 active at pChEMBL>=8 (38.3%). `src/pull_abl1.py`
- **QSAR reward model** (`src/qsar_model.py`): ECFP4 + RF, scaffold-split
  ROC-AUC 0.906 / PR-AUC 0.851; 5-fold CV ROC-AUC 0.952.
  → `checkpoints/abl1_rf.joblib`

## Decisions resolved (2026-08-24)
1. **Reward-model labeling: switched median→max aggregation, retrained.**
   `src/pull_abl1.py` now takes `max(pChEMBL)` per compound instead of the
   median (kept as `pchembl_median` field for reference); reasoning and
   trade-offs are in the file's docstring. Re-pulled from ChEMBL
   (3,097 compounds, 1,265 active at pChEMBL>=8, 40.8% vs old 38.3%) and
   retrained the RF (`src/qsar_model.py`, unchanged). Old median-based
   `abl1_qsar.json`/`abl1_rf.joblib`/`qsar_metrics.json` backed up at
   `backups/median_agg_2026-08-24/` on the pod. New RF: scaffold-split
   ROC-AUC 0.900, 5-fold CV ROC-AUC 0.948 (both close to the old model's).
   **imatinib now 0.736, dasatinib 0.932, nilotinib 0.726 — all three label
   active.** Binary classification (not regression) was kept deliberately:
   dynamic threshold shaping (still to implement) depends on a clean
   achievement signal.
2. **Reward hacking / applicability domain: added an RF-uncertainty penalty,
   not a hard Tanimoto cutoff.** `src/reward_model.py` computes
   `reward = P(active) - lambda_unc * std_over_trees(P(active))` (500-tree
   ensemble disagreement as the uncertainty proxy), rather than zeroing
   reward outside a similarity radius — a hard AD cutoff would fight the
   explicit goal of finding new scaffolds, whereas the uncertainty penalty
   only discourages regions where the forest itself doesn't agree, still
   letting confident novel structures score well. Applicability-domain
   Tanimoto-to-train-set is tracked as a diagnostic metric only (not used to
   shape reward), reported per step: `ad_pct_top_decile` = % of the batch's
   top-reward decile with max-Tanimoto-to-train >= 0.3.

## RL baseline — three runs, v3 is the working one
Collapse postmortems (v1, v2) and the fixes are documented in detail in the
`src/train_rl.py` module docstring. Short version:
- **v1** (`checkpoints/rl_baseline/`, REINFORCE, fixed beta_kl=0.02, reward
  floor-clipped at 0): collapsed at step ~750 to 2 molecules, stayed there
  for 4200+ steps. Cause: floor-clipping tied ~every early sample at
  reward 0, so the first real hit had enormous advantage and one uncapped
  gradient step swallowed the batch.
- **v2** (`checkpoints/rl_baseline_v2/`, + unclipped reward + threshold
  shaping): collapsed at step ~1900 — 2.5x later, but same endpoint. The
  runaway completes in ~40-60 steps, faster than the 50-step tau update
  cadence. The converged molecule was an EXACT match (Tanimoto 1.000) to
  the most potent compound in the ABL1 set (pChEMBL 10.52) — i.e. NOT
  reward hacking, the RF was correctly confident; the failure was purely
  optimization dynamics.
- **v3** (`checkpoints/rl_baseline_v3/`, + PPO-clipped multi-epoch updates
  + adaptive KL controller + dropout-off fix): **ran all 5000 steps with no
  collapse.** Visible self-correction at step ~1820: unique% dipped to 76%,
  beta_kl auto-ramped 0.003 -> 13.3 within 20 steps, suppressed the runaway,
  unique% recovered to 100%. Same again ~4660. 38.8 min on the 4090.

### v3 results (`src/eval_policy.py`, 5000 samples, vs pretrained prior)
| metric | prior | RL v3 |
|---|---|---|
| valid | 97.9% | **99.6%** |
| unique (of valid) | 100% | 96.0% |
| novel (of unique) | 94.2% | 89.1% |
| unique scaffolds | 4323 | 2938 |
| P(active) >= 0.5 | 0.04% | **5.7%** |
| P(active) >= 0.8 | 0.00% | **3.8%** |
| mean RF uncertainty | 0.284 | **0.223** |
| AD: top-decile within 0.3 | 84.8% | **94.4%** |

Reward went up ~140x at the P>=0.5 threshold while keeping 96% uniqueness
and 2938 distinct scaffolds — the thing v1/v2 could not do. Uncertainty
*fell* and AD-coverage *rose*, so the gain is not off-distribution
exploitation.

**Caveat — the win is partial. This is the headline limitation.**
Global diversity is healthy, but the high-reward tail is one chemotype with
R-group variation. Per `src/analyze_scaffolds.py` (5000 samples, two runs):

    unique molecules with P(active)>=0.5   118-125
    unique Bemis-Murcko scaffolds          33-42
    NOVEL (in neither ChEMBL train nor
      the ABL1 activity set)               116/118  (98%)
    top scaffold's share of the set        58-67%
    max-Tanimoto to ChEMBL train           mean 0.46, min 0.37, max 0.70

Read those together and the honest conclusion is: the molecules are
**novel** (98% appear in neither training set) and sit **inside** the
applicability domain (Tanimoto 0.37-0.70, so interpolation not
hallucination) — but they are **not scaffold-diverse**. Essentially every
high-reward scaffold is the same
`O=C(Nc1cc2cc(<Ar>)ccc2cn1)C1CC1` core — a cyclopropanecarboxamide on a
2-aminoquinoline/naphthyridine — with a different aryl group at one
position. That is **R-group enumeration around a single chemotype, not
scaffold hopping**, and only the latter is what "propose new scaffolds"
means. It is also the *same series* v1/v2 collapsed onto, now explored
around rather than converged onto.

Diversity-filtered experience replay is the next component and the direct
fix: penalize/skip rewarding a molecule whose scaffold is already
well-represented in the replay buffer, forcing the policy to look
elsewhere for advantage.

## v4 — diversity-filtered experience replay (RUN + ABLATED 2026-08-29)

Implemented and ablated 2026-08-29 on pod `rl-chemistry-v4`, 3x5000 steps,
~1.1 h wall clock (three runs concurrent; RDKit scoring is the bottleneck,
GPU sat at 9%).

### The baseline was wrong, and had to be recomputed first
`results/rl_baseline_v3/scaffolds.json` reports 33 Murcko scaffolds / 58%
top share. On the **generic-framework** key that the filter actually buckets
on, v3 is **19 frameworks with the top one covering 77%** — Murcko was
splitting phenyl/2-pyridyl/3-pyridyl variants of one core (67+10+8 = 85 of
117 molecules) into three "distinct scaffolds". Recomputed baseline is in
`results/rl_baseline_v3/scaffolds_framework.json`. Comparing v4 frameworks
against v3 Murcko would have measured the metric change, not the model.

### Results (n=5000, P(active)>=0.5 set; prior column for scale)
| metric | prior | v3 | v4a filter | v4b replay | v4c both |
|---|---|---|---|---|---|
| unique high-reward molecules | - | 117 | **411** | 5 | 187 |
| Murcko scaffolds | - | 32 | 359 | 4 | 161 |
| **generic frameworks** | - | **19** | **231** | 3 | **138** |
| **top framework share** | - | **77%** | **4%** | 40% | **7%** |
| novel (of high-reward) | - | 98% | 97% | 40% | 92% |
| P(active)>=0.5 | 0.10% | 5.7% | **11.4%** | 0.12% | 3.8% |
| P(active)>=0.8 | 0.02% | 3.8% | **8.9%** | 0.00% | 1.8% |
| valid% | 97.9 | 99.6 | 99.7 | 99.7 | 99.5 |
| unique% | 100 | 96.0 | 93.3 | 99.3 | **99.1** |
| overall scaffolds | 4321 | 2938 | 2753 | 2890 | **3073** |
| mean RF uncertainty | 0.283 | 0.223 | 0.199 | 0.237 | 0.224 |
| **AD: top-decile within 0.3** | 83.0 | **94.4** | **45.2** | 93.4 | **94.4** |

### Read this before quoting the v4a numbers
**v4a's diversity and potency gains are partly extrapolation.** Its
applicability-domain coverage halved (94.4% -> 45.2%); the high-reward set's
mean max-Tanimoto to train fell 0.459 -> 0.330 with a minimum of 0.252,
i.e. below the AD radius entirely. The RF-uncertainty penalty did **not**
catch it — uncertainty went *down* (0.199). That is a known random-forest
behaviour: tree agreement can stay high in regions the forest never saw, so
low ensemble variance is not evidence of validity out of domain. Half of
v4a's "better" molecules sit where the reward model has no standing to judge.

**v4c is the defensible result.** 7.3x more frameworks (19 -> 138), top-share
77% -> 7%, while holding AD coverage at *exactly* v3's 94.4%, uncertainty at
v3's level (0.224 vs 0.223), improving overall scaffold count (3073 vs 2938)
and uniqueness (99.1% vs 96.0%). The cost is potency: P>=0.5 falls 5.7% ->
3.8%. That is the honest trade — ~34% of the hit rate for 7.3x the chemotype
diversity, all of it inside the domain where the RF is trustworthy.

### Why replay alone (v4b) destroyed the run — the tau latch, confirmed
v4b was the only run without `--threshold-decay`, i.e. with v3's monotonic
tau, and it is the predicted failure exactly:
- A reward spike near step 1300 (unique% dipped to 64%) ratcheted tau up;
  it reached **0.953 at step 2950 and could never come down**.
- `pct_above_tau` was **0.0% for the entire last 1000 steps** — no molecule
  could clear the bar, so advantage variance collapsed and the run stopped
  learning. It did not look like collapse; the policy stayed 99% unique.
- Meanwhile the replay likelihood term ran unopposed: `beta_kl` pinned at
  its 20.0 ceiling for **798 steps** (v3: 14), losses swung -236 to +303,
  valid% crashed to 53%.
- Final: **P>=0.5 = 0.12% against the pretrained prior's own 0.10%**, and
  mean p_active 0.068 *below* the prior's 0.098. Worse than no RL at all.
- Replay barely functioned alone: 598 molecules admitted of 625,269 offered
  (0.1%), buffer 524 molecules over 262 scaffolds. In v4c the same buffer
  holds 1000 molecules over **1000 distinct scaffolds**, one per scaffold.

### The mechanism worth remembering
Replay alone is destructive; the filter alone drifts out of domain; together
they are complementary. The filter *pushes* the policy off the mined
chemotype, and the scaffold-stratified buffer *anchors* it to molecules
actually observed to score well — which is what keeps v4c inside the
applicability domain (94.4%) while v4a, with the same push and no anchor,
falls to 45.2%. The buffer is not mainly a sample-efficiency device here;
it is the thing that stops the diversity filter from wandering into
chemistry the reward model cannot evaluate.

Stability note: the filter *improved* optimisation stability rather than
harming it. v4a/v4c minimum unique% was 56.7% vs v3's 21.3%, minimum valid%
93.8/91.4 vs 82.0, and neither hit the beta_kl ceiling once (v3: 14 steps).
Removing the single dominant reward hill removed the pressure that was
driving v3 toward collapse.

- `src/diversity_filter.py` — per-scaffold occupancy memory. Reward
  multiplier `clamp(1 - count[key]/bucket_size, 0, 1)`, applied to the
  positive part of reward only.
- `src/replay_buffer.py` — scaffold-capped memory of high-reward molecules
  plus a char-level `encode()` (inverse of the sampler's decode).
- `src/train_rl.py` — both wired in behind `--diversity-filter` / `--replay`,
  **default off**, so the v3 command above still reproduces v3 exactly.
- `src/analyze_scaffolds.py` — now also reports generic-framework counts.

Three design calls that are the substance of the component (all three were
load-bearing in the results above):

1. **The filter keys on the generic framework, not the Murcko scaffold.**
   v3's top three "distinct" scaffolds are phenyl / 2-pyridyl / 3-pyridyl
   on one core. A Murcko-keyed filter is evadable by moving one ring
   nitrogen, and would make the headline metric improve while nothing
   chemically changed. `MakeScaffoldGeneric` merges them.
   **This also means v3's "33 scaffolds / 58% top share" is not the real
   baseline** — the generic-framework numbers for v3 have to be recomputed
   before any comparison, and they will look considerably worse.
2. **Filtered reward decays to 0.0, not to a floor.** Reward here is
   deliberately unclipped (prior samples sit near -0.17, invalid at -1.0),
   so REINVENT's score->0 convention would *promote* a filtered molecule
   above ordinary chemistry.
3. **Replay is an auxiliary per-token likelihood term, not extra PPO data.**
   Stale `logP_old` breaks the importance ratio; recomputing it makes
   ratio==1 and silently reverts to the uncapped REINFORCE step that
   collapsed v1.

### Risks predicted before the run — outcome
- **tau latch** — predicted as most likely failure. **Happened, in v4b,
  the one run without `--threshold-decay`.** Always set it when the filter
  or replay is on. Watch `tau` vs `reward_eff_mean` and `pct_above_tau`.
- **Replay as a collapse driver** — confirmed. Alone it pinned beta_kl at
  the ceiling for 798 steps and crashed valid% to 53%. Safe only alongside
  the filter, which keeps the buffer scaffold-diverse.
- **Filter too aggressive** — did *not* happen at `--df-bucket-size 25`;
  potency went up (v4a) or fell modestly (v4c). The real cost showed up
  somewhere unpredicted: **applicability domain, not potency** (v4a).
- `reward_raw_mean` vs `reward_eff_mean` logging was necessary — the gap
  between them is the only way to tell the filter working from the policy
  degrading.

### Ablation matrix (all vs. v3, 5000 steps each) — as run
| run | flags |
|---|---|
| v3 (baseline, re-eval on framework key) | *(none)* |
| v4a filter only | `--diversity-filter --threshold-decay 0.1` |
| v4b replay only | `--replay` *(no decay — this is why it failed)* |
| v4c both | `--diversity-filter --replay --threshold-decay 0.1` |

    nohup python src/train_rl.py --steps 5000 --batch 128 --lr 1e-4 \
      --beta-kl 0.02 --target-kl 3.0 --ppo-epochs 4 --clip-eps 0.2 \
      --lambda-unc 1.0 --threshold-update-every 50 \
      --diversity-filter --df-bucket-size 25 --df-mode generic \
      --replay --replay-coef 0.05 --replay-k 24 \
      --threshold-decay 0.1 \
      --out checkpoints/rl_v4c > logs/rl_v4c.log 2>&1 &

## How to run / reproduce
    # training (detached; ~39 min for 5000 steps on the 4090)
    nohup python src/train_rl.py --steps 5000 --batch 128 --lr 1e-4 \
      --beta-kl 0.02 --target-kl 3.0 --ppo-epochs 4 --clip-eps 0.2 \
      --lambda-unc 1.0 --threshold-update-every 50 \
      --out checkpoints/rl_baseline_v3 > logs/rl_baseline_v3.log 2>&1 &
    # evaluation vs the pretrained prior
    python src/eval_policy.py --policy checkpoints/rl_baseline_v3/policy_latest.pt --n 5000
Watch during a run: `unique_pct` + `scaffold_ratio` (collapse signal),
`beta_kl` (should spike when kl rises — that's the controller working),
`pct_above_tau` and `p_act` (learning signal). Per-step metrics land in
`<out>/history.json`, eval summary in `<out>/eval.json`.

## v5 — transfer learning + the complete 2^3 factorial (2026-08-29)

`transfer_phase()` in train_rl.py implements paper component 1: every
`--tl-every` steps the RL objective is suspended and the policy runs pure MLE
on a scaffold-diverse slice of the replay buffer (`ReplayBuffer.sample_diverse`,
round-robin over scaffolds so truncation costs breadth last). It shares the
buffer with the replay term but is a different mechanism — replay is an
auxiliary loss inside every update, permanently bounded by the PPO surrogate
it competes with; a TL phase has no RL objective present at all. That makes it
better at consolidating a rare chemotype and by far the most dangerous of the
three, since nothing inside an MLE phase bounds how far the policy moves.
Guards: smaller `--tl-lr`; a deduplicated scaffold-diverse training set; and
**a KL-to-prior tripwire after every epoch** (`--tl-max-kl`) that aborts the
phase mid-way. The tripwire is the load-bearing one — without it the adaptive
beta_kl controller only sees the damage on the *next* rollout, after the phase
has finished moving the policy. Over 31 phases across four runs it fired once.

### Full factorial, n=5000. F = diversity filter, R = replay, T = transfer learning
| run | FRT | frameworks | top fw | hi-reward mols | P>=.5 | P>=.8 | valid | uniq | scaffolds | unc | AD% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3   | `---` | 19 | 77% | 118 | 5.72 | 3.76 | 99.58 | 96.00 | 2938 | 0.223 | 94.4 |
| v4a  | `F--` | 231 | 4% | 411 | 11.36 | **8.89** | 99.66 | 93.34 | 2753 | 0.199 | **45.2** |
| v4b  | `-R-`* | 3 | 40% | 5 | 0.12 | 0.00 | 99.68 | 99.28 | 2890 | 0.237 | 93.4 |
| v4b2 | `-R-` | 7 | 14% | 7 | 0.10 | 0.04 | 99.68 | 99.68 | 3202 | 0.241 | 93.6 |
| v5a  | `--T` | 19 | 47% | 77 | 5.66 | 3.91 | 99.68 | 90.33 | 2105 | 0.192 | 97.0 |
| v4c  | `FR-` | 138 | 7% | 187 | 3.76 | 1.79 | 99.54 | 99.08 | 3073 | 0.224 | 94.4 |
| v5b  | `F-T` | 192 | 7% | 271 | 5.94 | 2.87 | 99.72 | 97.69 | 2801 | 0.210 | 96.6 |
| v5c  | `-RT` | 8 | 22% | 9 | 0.10 | 0.00 | 99.62 | 99.50 | 3101 | 0.243 | 93.4 |
| **v5d** | **`FRT`** | **333** | **10%** | **589** | **14.92** | 6.96 | **99.76** | 93.46 | 2542 | 0.215 | **99.8** |

\* v4b ran with monotonic tau (no `--threshold-decay`); v4b2 is the corrected re-run.

### What the factorial actually shows
1. **The diversity filter is the only component that produces learning +
   diversity on its own.** Every cell containing F reaches P>=0.5 of 3.8–14.9%
   and 138–333 frameworks. Every cell without F either fails to learn at all
   (any cell with R and no F) or learns v3-like potency with v3-like narrowness
   (v5a: 19 frameworks, 47% top share).
2. **Replay alone does not work, and the tau latch was not the reason.**
   This corrects the v4 write-up. v4b2 fixes the monotonic-tau confound and
   still reaches P>=0.5 = **0.10%**, against the pretrained prior's own ~0.10%.
   Replay is a *consolidation* mechanism: it re-presents molecules already
   found to be good, so with nothing generating those molecules it has nothing
   to consolidate and cannot bootstrap. Admission needs raw reward >= 0.4, and
   a run that never gets there never fills the buffer. v5c (`-RT`) confirms it:
   two consolidation mechanisms together, still 0.10%.
3. **All three together are superadditive, not merely additive.** v5d beats
   the best single component on diversity (333 vs 231 frameworks) *and* on
   potency (14.9% vs 11.4%) *and* fixes v4a's applicability-domain collapse
   outright (**99.8% vs 45.2%**, the highest AD of any run including v3).
4. **The AD story resolves cleanly.** F alone drifts out of domain (45.2%).
   Adding either memory mechanism repairs it — R (94.4%), T (96.6%), both
   (99.8%). This confirms the push/anchor account: the filter pushes the
   policy off the mined chemotype, and the buffer-backed mechanisms anchor it
   to chemistry already observed to score well. Their value here is domain
   tethering, not sample efficiency.

**Final model: `checkpoints/rl_v5d/policy_latest.pt`.**

## Candidate selection for docking
`src/select_candidates.py` — 5 generated candidates + 5 measured negatives,
in `results/candidates_v5d.json`. Not docked; selection only.

Two methodology notes that changed the output materially:
- **Negatives are size-matched on heavy-atom count** (candidates mean 32.6,
  all five negatives at 33). Docking scores are extensive in molecular size,
  so unmatched negatives would manufacture a positive result from arithmetic.
  They are measured-inactive ABL1 compounds (pChEMBL 4.02–4.78) rather than
  random or unmeasured decoys.
- **A framework count is a good aggregate statistic and a bad selection
  criterion.** The first run picked five "distinct framework" molecules that
  were all one chemotype, differing only in an amide substituent that happened
  to be cyclic — Murcko pulls substituent rings into the scaffold, so each got
  its own framework while the binding core was identical. At n=5 the metric is
  trivially gamed by decorating one core with different small rings. Fixed by
  adding a pairwise ECFP4 Tanimoto ceiling (`--max-sim 0.45`) on top.

## Docking (2026-08-30) — protocol validated, candidates INCOMPLETE

Tooling: smina (first pass), then **gnina v1.3.3** (Vina search + CNN
rescoring) on the pod; ligands 3D-embedded with ETKDG/MMFF and protonated at
**pH 7.4** with Dimorphite-DL (verified: dasatinib piperazine -> [NH+],
aspirin acid -> [O-]). `src/prep_receptors.py`, `src/build_benchmark.py`,
`src/run_docking.py`, `src/run_benchmark_docking.py`, `src/analyze_docking.py`.

### Receptor ensemble, and the AlphaFold finding
ABL1 inhibitors split by required conformation: type I (dasatinib) needs
DFG-in, type II (imatinib, nilotinib) needs DFG-out, whose allosteric back
pocket does not exist in DFG-in. Ensemble = 1IEP (imatinib, DFG-out), 3CS9
(nilotinib, DFG-out), 2GQG (dasatinib, DFG-in), AF-P00519 v6 (AlphaFold).

**The AlphaFold model is DFG-in.** Geometric test (DFG-Phe382 to alphaC-Glu286
and to Lys271):

    structure                    F382-E286  F382-K271   conformation
    1IEP  (imatinib)                 13.91      10.86   DFG-out
    3CS9  (nilotinib)                13.84      11.12   DFG-out
    2GQG  (dasatinib)                 8.78      14.04   DFG-in
    AlphaFold                         9.63      13.38   DFG-in

Kinase domain (242-495) excised from the 1130-residue model first: full-length
mean pLDDT is 64.7 with 49% of residues "very low", but the kinase domain
alone is **92.6**. CA-RMSD to 1IEP decomposes as whole domain 5.00 A,
N-lobe+hinge 2.07 A, post-A-loop 2.03 A — **the fold is right and essentially
all the error is in the activation loop / DFG region**, i.e. precisely what
governs type-II binding.

Confirmed independently by the known drugs (smina, exhaustiveness 16): the
penalty for docking against AlphaFold rather than the best crystal was
imatinib **+2.80**, nilotinib **+3.30** (both type II) vs dasatinib **+1.80**
(type I).

### Enrichment: 40 actives vs 40 property-matched inactives
`src/build_benchmark.py` — actives pChEMBL 9.70-10.82 drawn one per
Bemis-Murcko scaffold (40 distinct, controls analogue bias); inactives 4.21-
5.50 greedily 1:1 matched on heavy atoms. Residual imbalance **0.00 heavy
atoms, 0.03 logP** — docking scores are extensive in molecular size, so
without this the enrichment would be arithmetic rather than chemistry.

    scheme                       AUC    95% CI          EF10%
    affinity @ 1IEP  (DFG-out)  0.764  [0.652, 0.871]   1.50
    affinity @ 2GQG  (DFG-in)   0.698  [0.573, 0.816]   1.25
    affinity @ ensemble         0.776  [0.666, 0.873]   1.50
    CNNscore @ ensemble         0.783  [0.671, 0.884]   1.25
    CNNaffinity @ 2GQG          0.786  [0.677, 0.877]   2.00

Three conclusions:
1. **Discrimination is real but modest** — every CI excludes 0.5, but AUC
   tops out ~0.78. The earlier 5-negative control gave 0.733 on 15 pairs with
   a CI of [0.40, 0.95]: uninterpretable, which is why it was expanded.
2. **CNN rescoring did not help** (0.783 vs 0.776, indistinguishable). One of
   the two proposed fixes simply did not deliver.
3. **Docking is a WEAKER classifier than the QSAR RF already in hand**
   (~0.78 vs scaffold-split 0.900). Its value is *independence* — structure-
   based, different failure modes — not accuracy. Do not treat it as the more
   authoritative judge.
4. **An AlphaFold-only protocol would have been useless**: AUC 0.400 on the
   first control, i.e. worse than random, with actives and inactives separated
   by 0.07 kcal/mol.

### FINAL RESULT — candidates vs imatinib (exhaustiveness 16, `results/docking/final_table.json`)
All ligands docked in one run under an identical protocol, so imatinib is a
like-for-like anchor rather than a literature value. Score = best over the
three crystals; percentile is against the 40 known actives docked alongside.

    benchmark, same protocol:  40 known actives   mean -11.28  [-13.18, -8.82]
                               40 matched inactives mean -9.92  [-12.60, -5.77]

    molecule    source          dock   %ile vs actives  CNNaff     AF   AFpen   RF P
    nilotinib   marketed drug  -13.62       100%         8.869   -9.35  +4.27    -
    cand5       RL v5d         -12.97        95%         8.427  -10.59  +2.38   0.814
    imatinib    marketed drug  -12.79        92%         8.435   -8.03  +4.76    -
    cand3       RL v5d         -12.10        80%         7.765   -8.62  +3.48   0.863
    cand4       RL v5d         -11.39        50%         8.264   -7.77  +3.62   0.818
    cand1       RL v5d         -10.51        22%         7.826   -7.89  +2.62   1.000
    dasatinib   marketed drug  -10.32        12%         8.071   -9.33  +0.99    -
    cand2       RL v5d          -9.30         2%         7.214   -3.66  +5.64   0.985

- **The candidate set scores at the known-actives mean**: candidates mean
  -11.25 vs actives -11.28 vs matched inactives -9.92. As a distribution they
  land with the drugs, not the non-binders. That is the defensible claim.
- **One candidate (cand5, -12.97) outscores imatinib (-12.79)**; two of five
  sit above the 80th percentile of known actives. Do NOT read this as "cand5
  is more potent than imatinib" — see the caveat below.
- **dasatinib scores 12th percentile.** A marketed, sub-nanomolar ABL1 drug
  lands near the bottom. This is the protocol's error bar made visible, and
  the single best argument against over-reading any individual number.

### The RF and docking disagree, and that is the point of the exercise
Ranked by RF confidence the candidates are cand1 > cand2 > cand3 > cand4 >
cand5; ranked by docking they are cand5 > cand3 > cand4 > cand1 > cand2 —
close to exactly inverted. The RF's two most confident molecules
(P=1.000, 0.985) dock at the 22nd and 2nd percentile; its least confident
(P=0.814) docks at the 95th.

Neither model is thereby proven wrong: the RF is the stronger classifier on
its own benchmark (scaffold-split ROC-AUC 0.900 vs docking's 0.799), but it
judges 2D fingerprint similarity to known ABL1 chemistry, while docking judges
3D shape/chemical complementarity to a specific receptor conformation. They
fail differently, which is exactly why an orthogonal check was worth running.
The honest summary is that **no molecule here is corroborated by both methods
simultaneously**, and the candidates worth prioritising experimentally are the
ones that are at least not contradicted — cand3 and cand4, which sit mid-to-
high on both.

### Enrichment at exhaustiveness 16 (supersedes the exh-8 numbers)
    affinity @ 3CS9      0.799  [0.686, 0.903]   EF10% 1.50
    affinity @ ensemble  0.790  [0.680, 0.889]   EF10% 1.50
    affinity @ 2GQG      0.779  [0.667, 0.877]   EF10% 1.75
    affinity @ 1IEP      0.755  [0.641, 0.863]   EF10% 1.50
    cnn_score @ ensemble 0.778  [0.665, 0.883]
    cnn_affinity @ 2GQG  0.754  [0.640, 0.857]
CNN rescoring again failed to beat plain Vina scoring (0.778 vs 0.790).

### Search reliability — resolved, and why it mattered
At exhaustiveness 8, gnina returned **positive affinities** (+16.1, +129.8) for
one candidate, and imatinib scored **-4.42** against 1IEP, its own crystal
structure. At exhaustiveness 16 the same molecule scores **-12.79**. That is a
pure search failure, not chemistry, and it is why no candidate number from the
exhaustiveness-8 pass was ever reported. QC found only 2/160 benchmark runs
affected (1 active, 1 inactive), so the exh-8 AUC was not corrupted, but
exhaustiveness 8 is not usable for large flexible ligands on this target.
**Docking imatinib into its own crystal structure is the cheapest available
protocol sanity check — run it before trusting any batch.**

### Funds incident (2026-08-30, resolved)
The account balance hit zero mid-run and RunPod terminated the pod without
warning. Volume `e4akonl0eu` survived intact. Local safety copies of the three
irreplaceable artefacts (`rl_v5d/policy_latest.pt`, `generator_best.pt`,
`abl1_rf.joblib`, ~88 MB) now live in `checkpoints_backup/`, gitignored.
Keep them: the volume remains the only other copy.

## Not started
- Multi-seed docking (3+ seeds per ligand, take median) to put an error bar on
  each score. dasatinib's 12th-percentile result shows single-seed variance is
  large enough to matter.
- Rescoring the shortlist with a method that models the receptor as flexible
  (MM-GBSA or short MD), which is the standard escalation when empirical
  scoring saturates around AUC 0.8.
- Final write-up.
  Note this overlaps the replay term: replay is a likelihood term on
  remembered high-reward molecules inside the RL update, whereas the
  paper's component 1 is a separate periodic fine-tuning phase.
- Ablations of each component against the v3 baseline.
- Final candidate generation + write-up.

## Next step
Model work is complete: **v5d meets every success criterion set in advance**
(17.5x more generic frameworks, top share 77% -> 10%, potency 5.7% -> 14.9%,
AD 94.4% -> 99.8%, validity and novelty held). Remaining:
1. Agree a docking protocol, then score the 5 candidates vs the 5 matched
   negatives in `results/candidates_v5d.json`. **Do not run docking before
   that discussion.**
2. Final write-up.

The main known weakness is no longer the generator — it is the **reward
model**. The RF is the only judge of activity, was trained on 3,097 ABL1
compounds, and its ensemble variance provably fails to detect out-of-domain
drift (v4a: uncertainty fell to 0.199 while AD coverage halved). Docking is
valuable here precisely because it is an orthogonal check that does not
depend on the RF at all.

## Local mirror for review
The pod is the source of truth for data/weights, but all code + metrics are
mirrored to `C:\Users\Adam\rl_chemistry` for VS Code review:
`src/*.py` (12 scripts, md5-verified against the pod), `results/` (QSAR
metrics, v3 history/eval/scaffolds), `logs/` (all three RL runs),
`README.md`, `bootstrap.sh`, `requirements-lock.txt`. Weights, ChEMBL data
and the venv are NOT mirrored (too large) — they live only on the pod's
network volume. Transfer uses `<scratchpad>/rppush.sh` / `rppull.sh`
(scp is unavailable over the RunPod proxy; see memory `runpod-pod-access`).

## VS Code access
Remote-SSH alias `runpod-rlchem` is configured in
`C:\Users\Adam\.ssh\config` (see memory `runpod-pod-access`) — connect and
open `/workspace/rl_chemistry` to browse/watch files live. Its `User` line
must be updated whenever the pod is recreated, same as rp.sh's `POD=`.

## Pod lifecycle note
Pod is stopped between sessions (user's choice, to avoid idle billing).
Data/venv/checkpoints survive on the network volume. On resume: get the new
SSH string from the user (pod ID changes on restart/recreate), update
`POD=` in the rp.sh wrapper (see memory `runpod-pod-access`), verify with
a quick `nvidia-smi` + `import openchem` check before resuming work.
