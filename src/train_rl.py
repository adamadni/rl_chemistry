"""RL baseline: REINFORCE fine-tuning of the pretrained SMILES generator
against the ABL1 reward model, with dynamic reward-threshold shaping and a
KL-to-prior penalty against mode collapse.

Postmortem on the first baseline run (2026-08-24, beta_kl=0.02, no
threshold shaping, reward floor-clipped to 0): reward sat at ~0 for 750
steps (everything tied at the clip floor -> no gradient signal), then one
lucky sample broke through, REINFORCE massively overweighted it in a
single update (advantage = reward - baseline was enormous against a batch
of literal zeros), and the policy fully collapsed onto ~2 molecules for
the remaining 4200+ steps (unique% ~1.6-2.3%, kl_hat stuck ~12.5, reward
flat at 0.95 with loss ~0 -- baseline had caught up and there was nothing
left pushing exploration). Two independent fixes, both applied here:
  1. reward_model.py no longer floor-clips to 0 -- see that file. Early
     samples now get real negative-but-graded reward instead of a tied 0,
     so the first success isn't an infinite outlier by comparison.
  2. Dynamic threshold shaping (this file, ThresholdShaper): a
     monotonically non-decreasing bar tau, raised every `update_every`
     steps to the P-th percentile of recently-achieved raw reward, is
     subtracted from reward before the KL/baseline step.

Postmortem on the SECOND run (2026-08-24, same day, with fixes 1+2 above):
collapse still happened, just delayed ~2.5x (step ~1900 vs ~750). Traced
the exact converged molecule: it's an EXACT match (Tanimoto=1.000) to the
single most potent known compound in the ABL1 training set (pChEMBL 10.52),
sitting in a tight, genuinely potent SAR series -- so this is not reward
miscalibration or an RF blind spot, the forest is correctly and confidently
right (P=0.998, uncertainty=0.045). The runaway itself completes in ~40-60
steps once found (reward 0.06 -> 0.86 between two 20-step log points),
faster than ThresholdShaper's 50-step update cadence, so tau only catches
up *after* the batch has already concentrated on it, not during. Root cause
is mechanical, not reward-related: a single-epoch REINFORCE update lets one
batch's extreme-advantage sample dominate the ENTIRE gradient step, with
nothing bounding how far that one step can move the policy. Two more fixes,
both applied here:
  3. PPO-style clipped multi-epoch updates: each rollout (one sampled
     batch) is now optimized over `ppo_epochs` gradient steps using the
     importance ratio r = exp(logP_policy_new(seq) - logP_policy_old(seq))
     against the SAME advantage estimate, with the surrogate objective
     clipped to r in [1-eps, 1+eps]. This directly bounds how much
     probability mass any single rollout can shift toward one sequence,
     rather than taking one uncapped step sized by however large that
     rollout's advantage happened to be.
  4. Adaptive KL coefficient: beta_kl is no longer fixed. After each
     rollout, if the observed KL_hat exceeds target_kl by 1.5x, beta_kl is
     multiplied by 1.5 (and divided by 1.5 if well under target) that
     ramps the KL penalty up automatically exactly when a run starts
     drifting hard (kl_hat went ~1.5 -> 12.5 during both collapses),
     instead of relying on a single fixed value guessed in advance.
  Also fixed: dropout was left on (cfg's 0.2) during RL sampling AND during
  the PPO epochs' teacher-forced re-evaluation, so logP_policy differed
  between passes purely from random dropout masks, not just parameter
  updates -- meaningless noise in exactly the ratio PPO's clipping depends
  on. `load_generator` now forces dropout=0 unconditionally for RL.

v4 adds the paper's second stabilizer, diversity-filtered experience replay
(`--diversity-filter --replay`), which targets a DIFFERENT failure from
everything above. v1/v2 failed at *global collapse* -- the whole batch
converging to one or two molecules, unique% at 2% -- and PPO + the adaptive
KL controller fixed that. v3's remaining problem is not collapse: uniqueness
is 96% and the sample carries 2938 scaffolds. It is that the *high-reward
region* is one chemotype (58% of the P>=0.5 set on a single Bemis-Murcko
scaffold, and far more than that once ring-heteroatom variants are merged),
so the policy R-group-enumerates instead of scaffold-hopping. No amount of
trust-region control fixes this, because the objective contains no term that
distinguishes one scaffold from another -- the reward surface genuinely has
one dominant hill and PPO only governs the speed of the climb. The filter
changes the surface (diversity_filter.py); replay keeps earlier, rarer
chemotypes alive in the update so the policy is not re-paying their
discovery cost every time the trust region moves (replay_buffer.py). Both
default off; the CLAUDE.md v3 command line reproduces v3 unchanged.

v5 adds the paper's first stabilizer, transfer learning on the generator's
own high-reward output (`--transfer-learning`): every `--tl-every` steps the
RL objective is suspended and the policy does a few epochs of pure maximum
likelihood on a scaffold-diverse slice of the replay buffer. It shares the
buffer with the replay term but is a genuinely different mechanism -- replay
is an auxiliary loss competing with the PPO surrogate inside every update
and bounded by it, whereas a TL phase has no RL objective present at all.
That is why it consolidates a rare chemotype better, and why it is the most
dangerous of the three: nothing inside an MLE phase bounds how far the policy
moves. The KL-to-prior tripwire in `transfer_phase` is the real guard; the
smaller LR and the diverse training set only slow it down.

Objective per rollout:
    tau              = ThresholdShaper's current bar (monotonic)
    shaped_reward    = reward(seq) - tau
    KL_hat(seq)      = logP_old(seq) - logP_prior(seq)   [MC estimate, fixed for the rollout]
    beta_kl          = adaptive, updated once per rollout toward target_kl
    augmented_reward = shaped_reward - beta_kl * KL_hat(seq)
    advantage        = augmented_reward - baseline        [EMA baseline, fixed for the rollout]
  then for ppo_epochs steps on the SAME rollout:
    ratio  = exp(logP_policy(seq) - logP_old(seq))
    loss   = -mean(min(ratio*advantage, clip(ratio, 1-eps, 1+eps)*advantage))

`prior` is a frozen copy of the pretrained checkpoint, loaded once and never
updated.

Diagnostics tracked every log step (not used to shape reward except tau
itself, which is deliberately part of the objective):
  valid%, unique%, unique-scaffold ratio (mode-collapse proxies)
  mean reward / p_active / RF-uncertainty, tau, %-of-batch-above-tau
  mean max-Tanimoto to a train-set reference sample, and % of the batch's
  top-reward decile within an applicability-domain radius (Tanimoto>=0.3)
  KL_hat (drift from prior), beta_kl (adaptive coefficient's current value)
"""
import argparse, collections, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/workspace/rl_chemistry/src")
from smiles_utils import canonicalize
from reward_model import RewardModel
from diversity_filter import DiversityFilter, scaffold_key
from replay_buffer import ReplayBuffer, encode
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
DATA = "/workspace/rl_chemistry/data/processed"
CKPT = "/workspace/rl_chemistry/checkpoints"
GEN_CKPT = f"{CKPT}/generator_best.pt"
DEV = torch.device("cuda")
class SmilesRNN(nn.Module):
    def __init__(self, vsize, emb, hid, layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vsize, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hid, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, vsize)
    def forward(self, x, h=None):
        e = self.emb(x)
        o, h = self.rnn(e, h)
        return self.fc(self.drop(o)), h
def load_generator(trainable):
    """dropout forced to 0 regardless of `trainable` -- RL needs logP_policy
    to be a deterministic function of parameters + sampled tokens (dropout
    noise would make the PPO ratio exp(logp_new - logp_old) reflect random
    mask differences, not the actual parameter update)."""
    ck = torch.load(GEN_CKPT, weights_only=False)
    cfg = ck["config"]
    vocab = ck["vocab"]
    m = SmilesRNN(len(vocab), cfg["emb"], cfg["hid"], cfg["layers"], dropout=0.0).to(DEV)
    m.load_state_dict(ck["model"])
    m.train(trainable)
    for p in m.parameters():
        p.requires_grad_(trainable)
    return m, vocab, cfg
def sample_with_logprobs(policy, vocab, batch, max_len, temp=1.0):
    """Autoregressive sampling WITH gradient-tracked policy log-probs.
    Returns: token_ids (batch, L) int64 padded with PAD, seq_logp (batch,)
    summed log pi(a_t|s_<t) over the generated (non-pad) tokens including EOS,
    and the decoded SMILES strings (empty string if EOS never emitted).
    """
    PAD, BOS, EOS = vocab["pad"], vocab["bos"], vocab["eos"]
    itos = {i: s for s, i in vocab.items() if s not in ("pad", "bos", "eos")}
    x = torch.full((batch, 1), BOS, dtype=torch.long, device=DEV)
    h = None
    done = torch.zeros(batch, dtype=torch.bool, device=DEV)
    seq_logp = torch.zeros(batch, device=DEV)
    tokens = [[] for _ in range(batch)]
    all_ids = []
    for _ in range(max_len):
        logits, h = policy(x, h)
        logits = logits[:, -1, :] / temp
        logits = logits.clone()
        logits[:, PAD] = -1e9
        logits[:, BOS] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        nxt = dist.sample()
        lp = dist.log_prob(nxt)
        seq_logp = seq_logp + torch.where(done, torch.zeros_like(lp), lp)
        nxt_list = nxt.tolist()
        for i, t in enumerate(nxt_list):
            if not done[i]:
                tokens[i].append(t)
                if t != EOS:
                    pass
        newly_done = (nxt == EOS)
        done = done | newly_done
        all_ids.append(nxt)
        x = nxt.unsqueeze(1)
        if done.all():
            break
    smiles = []
    clean_ids = []
    for i, ids in enumerate(tokens):
        if len(ids) and ids[-1] == EOS:
            body = ids[:-1]
            smiles.append("".join(itos[t] for t in body if t in itos))
        else:
            smiles.append("")
        clean_ids.append([BOS] + ids)
    L = max(len(s) for s in clean_ids)
    padded = torch.full((batch, L), PAD, dtype=torch.long, device=DEV)
    for i, ids in enumerate(clean_ids):
        padded[i, :len(ids)] = torch.tensor(ids, device=DEV)
    return padded, seq_logp, smiles
def _teacher_forced_logprob(model, padded, vocab):
    """Shared implementation: teacher-forced sum_t logP(token_t | token_<t)
    for the given (already-sampled) token sequences. Differentiable w.r.t.
    `model`'s parameters when called outside no_grad -- used both for the
    frozen prior (wrapped in no_grad below) and for the trainable policy's
    per-PPO-epoch re-evaluation (grad enabled)."""
    PAD = vocab["pad"]
    logits, _ = model(padded[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    target = padded[:, 1:]
    tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != PAD).float()
    return (tok_logp * mask).sum(dim=1)
@torch.no_grad()
def prior_logprob(prior, padded, vocab):
    """Teacher-forced log P_prior(seq) for sequences sampled from the policy."""
    return _teacher_forced_logprob(prior, padded, vocab)
def policy_logprob(policy, padded, vocab):
    """Teacher-forced log P_policy(seq), WITH gradient -- used once per PPO
    epoch to re-evaluate the fixed rollout under the current (updating)
    policy parameters."""
    return _teacher_forced_logprob(policy, padded, vocab)
class ADReference:
    """Applicability-domain diagnostic: max-Tanimoto to a fixed sample of
    the training set. Diagnostic only -- not used to shape reward."""
    def __init__(self, n_ref=5000, radius=2, nbits=2048, seed=42):
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        with open(f"{DATA}/chembl_train.smi") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        rng = random.Random(seed)
        sample = rng.sample(lines, min(n_ref, len(lines)))
        fps = []
        for s in sample:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fps.append(self._gen.GetFingerprintAsNumPy(m))
        self.ref = np.array(fps, dtype=np.uint8)  # (n_ref, nbits)
        self.ref_sum = self.ref.sum(axis=1)
    def max_tanimoto(self, smiles_list, valid_mask):
        out = np.zeros(len(smiles_list), dtype=np.float32)
        for i, (s, v) in enumerate(zip(smiles_list, valid_mask)):
            if not v:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = self._gen.GetFingerprintAsNumPy(m).astype(np.uint8)
            inter = (self.ref & fp).sum(axis=1)
            union = self.ref_sum + fp.sum() - inter
            sims = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)
            out[i] = sims.max()
        return out
class ThresholdShaper:
    """Reward bar tau, raised every `update_every` steps to the P-th
    percentile of recent reward. Monotonically non-decreasing when
    `decay`=0 (v3 behaviour). See module docstring for why this is the
    actual fix for a mode that "solved" the task and stopped exploring.

    `decay` > 0 lets tau ease back down toward the current percentile when
    achievement falls, and exists specifically because the diversity filter
    breaks the assumption monotonicity relies on. Monotonic tau is safe when
    the reward function is fixed: reward that was once achievable stays
    achievable, so a bar that only rises still has molecules above it. The
    filter deliberately destroys the reward of whatever region set the bar in
    the first place, so a latched tau can end up above *everything* the
    policy can now reach -- at which point every shaped reward in the batch
    is negative, advantage variance collapses toward zero, and the run stops
    learning rather than exploring elsewhere. A slow one-sided decay bounds
    how far tau can outrun the filtered reward landscape while still
    ratcheting up whenever the policy genuinely improves.
    """
    def __init__(self, batch, window_batches=20, percentile=70.0, update_every=50,
                 init=0.0, decay=0.0):
        self.buf = collections.deque(maxlen=window_batches * batch)
        self.percentile = percentile
        self.update_every = update_every
        self.decay = decay
        self.tau = init
    def update(self, step, raw_rewards):
        self.buf.extend(raw_rewards.tolist())
        if step % self.update_every == 0 and len(self.buf) >= self.update_every:
            candidate = float(np.percentile(np.asarray(self.buf), self.percentile))
            if candidate >= self.tau or self.decay <= 0:
                self.tau = max(self.tau, candidate)
            else:
                self.tau += self.decay * (candidate - self.tau)
        return self.tau
def replay_nll(policy, padded, vocab):
    """Mean per-token NLL of stored sequences under the current policy.

    Per-token rather than per-sequence: see replay_buffer.py's docstring --
    summed sequence logP is O(-40) against a PPO surrogate of O(0.1-1), so
    the un-normalised version silently turns the run into supervised training
    on the buffer.
    """
    PAD = vocab["pad"]
    logits, _ = policy(padded[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    target = padded[:, 1:]
    tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != PAD).float()
    return -(tok_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
def transfer_phase(policy, prior, opt, buf, vocab, args, step):
    """Paper component 1: periodic supervised fine-tuning of the generator on
    its OWN high-reward output, between RL rollouts.

    Distinct from the replay term, which is an auxiliary loss *inside* each RL
    update and therefore always competing with the PPO surrogate and bounded
    by it. This is a separate phase: for a few hundred gradient steps the RL
    objective is not present at all and the policy is doing pure maximum
    likelihood on a fixed molecule set. That is what makes it effective at
    consolidating a chemotype the policy found only occasionally -- and what
    makes it by far the most dangerous of the three components. Nothing in an
    MLE phase bounds how much probability mass moves onto the training set:
    no clipping, no trust region, no advantage weighting. Run long enough it
    would simply overwrite the policy with the buffer.

    Three things bound it here, and the third is the one that actually matters:
      1. A smaller LR than RL (`--tl-lr`), applied by temporarily overriding
         the optimiser's LR rather than building a second Adam -- two Adams on
         the same parameters keep separate momentum estimates that then fight
         each other across phase boundaries.
      2. A scaffold-diverse, deduplicated training set from the replay
         buffer's round-robin sampler, so MLE cannot concentrate on one core.
      3. **A KL-to-prior tripwire checked after every epoch.** The frozen
         pretrained prior is the only fixed reference point in the whole
         system; if an MLE phase drags the policy further than `--tl-max-kl`
         from it, the phase aborts mid-way and RL resumes. Without this the
         adaptive beta_kl controller only sees the damage on the *next*
         rollout, i.e. after the phase has already finished moving the policy.

    Returns a dict of diagnostics; a no-op (empty dict) when the buffer has
    too few molecules to be worth a phase.
    """
    mols, w = buf.sample_diverse(args.tl_max_mols)
    if len(mols) < args.tl_min_mols:
        return {}
    padded, keep = encode(mols, vocab, args.max_len, DEV)
    if padded is None:
        return {}
    w = torch.tensor(w[keep], device=DEV, dtype=torch.float32)
    n = padded.shape[0]
    base_lr = opt.param_groups[0]["lr"]
    for g in opt.param_groups:
        g["lr"] = args.tl_lr
    policy.train()
    nll_last, epochs_run, aborted, kl_now = 0.0, 0, False, 0.0
    try:
        for epoch in range(args.tl_epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, args.tl_batch):
                idx = perm[i:i + args.tl_batch]
                nll = replay_nll(policy, padded[idx], vocab)
                loss = (w[idx] * nll).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()
                nll_last = float(loss.item())
            epochs_run = epoch + 1
            # ---- tripwire: how far has this phase pushed us from the prior?
            with torch.no_grad():
                probe, plogp_probe, _ = sample_with_logprobs(
                    policy, vocab, min(64, args.batch), args.max_len, args.temp)
                kl_now = float((plogp_probe - prior_logprob(prior, probe, vocab)).mean().item())
            if kl_now > args.tl_max_kl:
                aborted = True
                break
    finally:
        for g in opt.param_groups:
            g["lr"] = base_lr
    return {"tl_step": step, "tl_n_mols": n, "tl_epochs_run": epochs_run,
            "tl_nll": nll_last, "tl_kl_after": kl_now, "tl_aborted": aborted}
def scaffold_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=0.02, help="initial value; adapted online toward --target-kl")
    ap.add_argument("--target-kl", type=float, default=3.0)
    ap.add_argument("--beta-kl-min", type=float, default=1e-3)
    ap.add_argument("--beta-kl-max", type=float, default=20.0)
    ap.add_argument("--kl-adapt-factor", type=float, default=1.5)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--lambda-unc", type=float, default=1.0)
    ap.add_argument("--baseline-momentum", type=float, default=0.95)
    ap.add_argument("--threshold-window-batches", type=int, default=20)
    ap.add_argument("--threshold-percentile", type=float, default=70.0)
    ap.add_argument("--threshold-update-every", type=int, default=50)
    ap.add_argument("--threshold-init", type=float, default=0.0)
    ap.add_argument("--threshold-decay", type=float, default=0.0,
                    help="0 = monotonic tau (v3). >0 lets tau ease down when the "
                         "diversity filter suppresses the region that set it.")
    # ---- diversity filter (component 2a) -- all default OFF so the v3
    # ---- command line in CLAUDE.md still reproduces v3 exactly.
    ap.add_argument("--diversity-filter", action="store_true")
    ap.add_argument("--df-bucket-size", type=int, default=25)
    ap.add_argument("--df-mode", choices=["generic", "murcko"], default="generic")
    ap.add_argument("--df-record-threshold", type=float, default=0.5,
                    help="P(active) a molecule must reach to consume bucket capacity")
    # ---- experience replay (component 2b)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--replay-capacity", type=int, default=1000)
    ap.add_argument("--replay-max-per-scaffold", type=int, default=10)
    ap.add_argument("--replay-min-reward", type=float, default=0.4,
                    help="admission bar on RAW reward; scaffold concentration is "
                         "handled by the per-scaffold cap, not by this")
    ap.add_argument("--replay-k", type=int, default=24)
    ap.add_argument("--replay-coef", type=float, default=0.05)
    ap.add_argument("--replay-start", type=int, default=200)
    # ---- transfer learning on own high-reward output (component 1)
    ap.add_argument("--transfer-learning", action="store_true",
                    help="periodic supervised MLE phases on the buffer; populates the "
                         "buffer even when --replay (the inline term) is off")
    ap.add_argument("--tl-every", type=int, default=500)
    ap.add_argument("--tl-start", type=int, default=1000)
    ap.add_argument("--tl-epochs", type=int, default=2)
    ap.add_argument("--tl-batch", type=int, default=128)
    ap.add_argument("--tl-max-mols", type=int, default=500)
    ap.add_argument("--tl-min-mols", type=int, default=50,
                    help="skip the phase entirely below this many buffered molecules")
    ap.add_argument("--tl-lr", type=float, default=1e-5)
    ap.add_argument("--tl-max-kl", type=float, default=8.0,
                    help="abort the phase mid-way if KL-to-prior exceeds this")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", type=str, default=f"{CKPT}/rl_baseline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print("loading policy + frozen prior...", flush=True)
    policy, vocab, cfg = load_generator(trainable=True)
    prior, _, _ = load_generator(trainable=False)
    reward_model = RewardModel(lambda_unc=args.lambda_unc)
    ad_ref = ADReference()
    shaper = ThresholdShaper(args.batch, args.threshold_window_batches, args.threshold_percentile,
                             args.threshold_update_every, args.threshold_init, args.threshold_decay)
    dfilter = DiversityFilter(args.df_bucket_size, args.df_mode,
                              args.df_record_threshold) if args.diversity_filter else None
    # The buffer backs BOTH component 1 (transfer-learning phases) and
    # component 2's inline replay term, so either flag creates it.
    replay = ReplayBuffer(args.replay_capacity, args.replay_max_per_scaffold,
                          args.replay_min_reward, args.seed) \
        if (args.replay or args.transfer_learning) else None
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0
    beta_kl = args.beta_kl
    history = []
    print(f"config: {vars(args)}", flush=True)
    print(f"reward model scheme={reward_model.scheme} threshold={reward_model.threshold} "
          f"lambda_unc={reward_model.lambda_unc} invalid_reward={reward_model.invalid_reward}", flush=True)
    # Extra columns only appear when the component is enabled, so a v3-style
    # run's log stays byte-comparable with logs/rl_baseline_v3.log.
    extra_hdr = ""
    if dfilter is not None:
        extra_hdr += f"{'dfmult':>8}{'sat':>5}"
    if replay is not None:
        extra_hdr += f"{'buf':>6}{'bscaf':>7}{'rnll':>7}"
    print(f"\n{'step':>6}{'loss':>10}{'reward':>9}{'tau':>7}{'p_act':>8}{'unc':>7}{'kl':>8}{'beta':>7}"
          f"{'valid%':>8}{'uniq%':>7}{'scaf/uniq':>10}{'>tau%':>7}{'AD%(top)':>9}"
          f"{extra_hdr}{'sec':>7}", flush=True)
    t_start = time.time()
    for step in range(1, args.steps + 1):
        t0 = time.time()
        policy.train()
        padded, old_logp, smiles = sample_with_logprobs(policy, vocab, args.batch, args.max_len, args.temp)
        old_logp = old_logp.detach()
        with torch.no_grad():
            plogp = prior_logprob(prior, padded, vocab)
        canon = [canonicalize(s) if s else None for s in smiles]
        valid_mask = np.array([c is not None for c in canon])
        score_in = [c if c else "" for c in canon]
        scores = reward_model.score(score_in)
        raw_reward = scores["reward"]
        keys = None
        if dfilter is not None or replay is not None:
            keys = [scaffold_key(c, args.df_mode) if c else None for c in canon]
        # ---- Diversity filter, applied BEFORE the shaper so tau tracks what
        # ---- is achievable *under* the filter rather than the pre-filter
        # ---- landscape (a tau set by the unfiltered peak is unreachable once
        # ---- the filter suppresses that peak -- see ThresholdShaper).
        # ---- Multipliers are read against the bucket state at batch start and
        # ---- only recorded afterwards, so every molecule in a step is judged
        # ---- against the same memory and the result is order-independent.
        eff_reward = raw_reward
        df_mult_mean = 1.0
        if dfilter is not None:
            mult = np.asarray(dfilter.multipliers(keys), dtype=np.float32)
            eff_reward = np.where(raw_reward > 0, raw_reward * mult, raw_reward).astype(np.float32)
            dfilter.record(keys, scores["p_active"])
            df_mult_mean = float(mult.mean())
        if replay is not None:
            replay.add(canon, keys, raw_reward)
        tau = shaper.update(step, eff_reward)
        shaped_reward = torch.tensor(eff_reward - tau, device=DEV, dtype=torch.float32)
        kl_hat = (old_logp - plogp).detach()
        kl_mean = kl_hat.mean().item()
        if kl_mean > args.target_kl * args.kl_adapt_factor:
            beta_kl = min(beta_kl * args.kl_adapt_factor, args.beta_kl_max)
        elif kl_mean < args.target_kl / args.kl_adapt_factor:
            beta_kl = max(beta_kl / args.kl_adapt_factor, args.beta_kl_min)
        augmented = shaped_reward - beta_kl * kl_hat
        baseline = args.baseline_momentum * baseline + (1 - args.baseline_momentum) * augmented.mean().item()
        advantage = (augmented - baseline).detach()
        # ---- PPO-style clipped update: ppo_epochs gradient steps on this
        # ---- ONE rollout, ratio/clip bounding how far any single step moves.
        # ---- One scaffold-stratified draw from replay memory, fixed across
        # ---- this rollout's PPO epochs. Enters as an auxiliary likelihood
        # ---- term, never through the PPO ratio: a molecule stored thousands
        # ---- of steps ago has a stale logP_old, which either saturates the
        # ---- clip or (if recomputed) makes ratio==1 and silently reverts to
        # ---- the uncapped REINFORCE step that collapsed v1.
        rpad, rw = None, None
        if args.replay and replay is not None and step >= args.replay_start and len(replay):
            rs, w = replay.sample(args.replay_k)
            if rs:
                rpad, keep = encode(rs, vocab, args.max_len, DEV)
                if rpad is not None:
                    rw = torch.tensor(w[keep], device=DEV, dtype=torch.float32)
        last_loss = 0.0
        replay_term = 0.0
        for _ in range(args.ppo_epochs):
            new_logp = policy_logprob(policy, padded, vocab)
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
            surrogate = torch.min(ratio * advantage, clipped * advantage)
            loss = -surrogate.mean()
            if rpad is not None:
                rl = (rw * replay_nll(policy, rpad, vocab)).mean()
                loss = loss + args.replay_coef * rl
                replay_term = float(rl.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            last_loss = loss.item()
        loss = last_loss
        # ---- diagnostics (no grad) ----
        valid_pct = 100.0 * valid_mask.mean()
        uniq = list({c for c in canon if c})
        uniq_pct = 100.0 * len(uniq) / max(1, valid_mask.sum())
        scafs = {s for s in (scaffold_of(c) for c in uniq) if s is not None}
        scaf_ratio = len(scafs) / max(1, len(uniq))
        max_tan = ad_ref.max_tanimoto(score_in, valid_mask)
        order = np.argsort(-scores["reward"])
        topk = order[:max(1, len(order) // 10)]
        ad_pct_top = 100.0 * (max_tan[topk] >= 0.3).mean() if len(topk) else 0.0
        pct_above_tau = 100.0 * (scores["reward"] > tau).mean()
        dt = time.time() - t0
        row = {"step": step, "loss": loss, "reward_mean": float(scores["reward"].mean()),
               "tau": float(tau), "pct_above_tau": float(pct_above_tau), "beta_kl": float(beta_kl),
               "p_active_mean": float(scores["p_active"].mean()),
               "uncertainty_mean": float(scores["uncertainty"].mean()),
               "kl_hat_mean": float(kl_mean),
               "valid_pct": float(valid_pct), "unique_pct": float(uniq_pct),
               "scaffold_ratio": float(scaf_ratio), "ad_pct_top_decile": float(ad_pct_top),
               "baseline": baseline}
        # Raw vs effective reward are logged separately: their gap IS the
        # filter's bite, and `reward_mean` alone cannot distinguish "the
        # policy got worse" from "the filter is working as intended".
        row["reward_raw_mean"] = float(raw_reward.mean())
        row["reward_eff_mean"] = float(np.mean(eff_reward))
        if dfilter is not None:
            row["df_mult_mean"] = df_mult_mean
            row.update({f"df_{k}": v for k, v in dfilter.stats().items()})
        if replay is not None:
            row["replay_nll"] = replay_term
            row.update({f"replay_{k}": v for k, v in replay.stats().items()})
        # ---- Transfer-learning phase, AFTER this step's RL update so the
        # ---- rollout it was computed from is fully consumed first.
        if (args.transfer_learning and replay is not None and step >= args.tl_start
                and step % args.tl_every == 0):
            tl = transfer_phase(policy, prior, opt, replay, vocab, args, step)
            if tl:
                row.update(tl)
                print(f"      TL @ step {step}: {tl['tl_n_mols']} mols, "
                      f"{tl['tl_epochs_run']}/{args.tl_epochs} epochs, nll {tl['tl_nll']:.3f}, "
                      f"kl_after {tl['tl_kl_after']:.2f}"
                      f"{' ABORTED (kl tripwire)' if tl['tl_aborted'] else ''}", flush=True)
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            extra = ""
            if dfilter is not None:
                extra += f"{df_mult_mean:>8.3f}{dfilter.stats()['n_saturated']:>5d}"
            if replay is not None:
                rs_ = replay.stats()
                extra += f"{rs_['size']:>6d}{rs_['n_scaffolds']:>7d}{replay_term:>7.3f}"
            print(f"{step:>6}{loss:>10.4f}{row['reward_mean']:>9.3f}{tau:>7.3f}"
                  f"{row['p_active_mean']:>8.3f}{row['uncertainty_mean']:>7.3f}{kl_mean:>8.3f}{beta_kl:>7.3f}"
                  f"{valid_pct:>8.1f}{uniq_pct:>7.1f}{scaf_ratio:>10.3f}{pct_above_tau:>7.1f}"
                  f"{ad_pct_top:>9.1f}{extra}{dt:>7.2f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": policy.state_dict(), "vocab": vocab, "config": cfg,
                        "step": step, "args": vars(args)}, f"{args.out}/policy_step{step}.pt")
            torch.save({"model": policy.state_dict(), "vocab": vocab, "config": cfg,
                        "step": step, "args": vars(args)}, f"{args.out}/policy_latest.pt")
            json.dump(history, open(f"{args.out}/history.json", "w"), indent=1)
            examples = [c for c in canon if c][:5]
            print(f"      saved checkpoint @ step {step} | examples: {examples}", flush=True)
    total_min = (time.time() - t_start) / 60
    print(f"\nRL_BASELINE_DONE in {total_min:.1f} min", flush=True)
if __name__ == "__main__":
    main()
