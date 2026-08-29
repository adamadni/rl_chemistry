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

## Not started
- **A v4b re-run with `--threshold-decay 0.1`** — v4b's failure is confounded:
  it tested "replay alone" AND "monotonic tau" at once, so it does not
  cleanly isolate replay's contribution. Rerunning it with decay on is the
  one ablation still owed.
- **Deciding v4a vs v4c**, i.e. whether the AD collapse is acceptable. If
  extrapolation is the concern, the fix is a real AD term in the reward
  rather than relying on RF ensemble variance, which demonstrably fails
  out of domain.
- Transfer learning on own high-reward outputs (3rd paper component).
  Note this overlaps the replay term: replay is a likelihood term on
  remembered high-reward molecules inside the RL update, whereas the
  paper's component 1 is a separate periodic fine-tuning phase.
- Ablations of each component against the v3 baseline.
- Final candidate generation + write-up.

## Next step
The success criterion set before the run was: more generic frameworks AND a
lower top-framework share, while holding valid%, novelty, `pct_p_ge_0.5`
and the AD metrics. **v4c meets it on every axis except `pct_p_ge_0.5`**
(3.8% vs 5.7%); v4a beats it on potency and diversity but fails the AD
criterion badly (45.2% vs 94.4%). So:
1. Re-run v4b with `--threshold-decay 0.1` to disentangle "replay alone"
   from "monotonic tau" — currently confounded.
2. Decide the v4a/v4c trade (potency+diversity vs domain validity). If v4a's
   chemistry is wanted, first replace the RF-variance uncertainty penalty
   with an explicit AD term — ensemble variance provably did not detect
   the out-of-domain drift here.
3. Then: transfer learning on own high-reward outputs, and final candidate
   generation from whichever policy is chosen.

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
