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
    """Monotonically non-decreasing reward bar tau. Every `update_every`
    steps, tau is raised (never lowered) to the P-th percentile of raw
    reward over the last `window_batches` batches. See module docstring
    for why this is the actual fix for a mode that "solved" the task and
    stopped exploring."""
    def __init__(self, batch, window_batches=20, percentile=70.0, update_every=50, init=0.0):
        self.buf = collections.deque(maxlen=window_batches * batch)
        self.percentile = percentile
        self.update_every = update_every
        self.tau = init
    def update(self, step, raw_rewards):
        self.buf.extend(raw_rewards.tolist())
        if step % self.update_every == 0 and len(self.buf) >= self.update_every:
            candidate = float(np.percentile(np.asarray(self.buf), self.percentile))
            self.tau = max(self.tau, candidate)
        return self.tau
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
                             args.threshold_update_every, args.threshold_init)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0
    beta_kl = args.beta_kl
    history = []
    print(f"config: {vars(args)}", flush=True)
    print(f"reward model scheme={reward_model.scheme} threshold={reward_model.threshold} "
          f"lambda_unc={reward_model.lambda_unc} invalid_reward={reward_model.invalid_reward}", flush=True)
    print(f"\n{'step':>6}{'loss':>10}{'reward':>9}{'tau':>7}{'p_act':>8}{'unc':>7}{'kl':>8}{'beta':>7}"
          f"{'valid%':>8}{'uniq%':>7}{'scaf/uniq':>10}{'>tau%':>7}{'AD%(top)':>9}{'sec':>7}", flush=True)
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
        tau = shaper.update(step, scores["reward"])
        shaped_reward = torch.tensor(scores["reward"] - tau, device=DEV, dtype=torch.float32)
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
        last_loss = 0.0
        for _ in range(args.ppo_epochs):
            new_logp = policy_logprob(policy, padded, vocab)
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
            surrogate = torch.min(ratio * advantage, clipped * advantage)
            loss = -surrogate.mean()
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
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(f"{step:>6}{loss:>10.4f}{row['reward_mean']:>9.3f}{tau:>7.3f}"
                  f"{row['p_active_mean']:>8.3f}{row['uncertainty_mean']:>7.3f}{kl_mean:>8.3f}{beta_kl:>7.3f}"
                  f"{valid_pct:>8.1f}{uniq_pct:>7.1f}{scaf_ratio:>10.3f}{pct_above_tau:>7.1f}"
                  f"{ad_pct_top:>9.1f}{dt:>7.2f}", flush=True)
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
