"""Scaffold-capped replay memory of high-reward molecules.

Component 2 of the paper's three stabilizers. The buffer keeps molecules the
policy has already found to be good and re-presents them during later
updates, so a chemotype discovered at step 400 is not forgotten by step 4000
just because the policy has drifted -- sparse-reward RL otherwise pays the
full exploration cost again every time the trust region moves.

## Why replayed molecules do NOT go through the PPO ratio

The obvious implementation -- append replayed sequences to the rollout and
let the existing clipped surrogate handle them -- is wrong here. PPO's ratio
`exp(logP_new(seq) - logP_old(seq))` is only meaningful when logP_old came
from the behaviour policy that actually sampled `seq`. A molecule stored 2000
steps ago was sampled by a policy that no longer exists; its stored logP_old
is stale by an unbounded amount, so the ratio saturates the clip range
immediately and contributes either a zero gradient or a wildly mis-scaled
one. Recomputing logP_old under the *current* policy instead makes the ratio
identically 1 at the first epoch, which silently turns the clipped surrogate
back into plain un-clipped REINFORCE on those samples -- reintroducing the
exact uncapped-single-sample mechanism that collapsed v1.

So replay enters as a separate auxiliary likelihood term added to the PPO
loss, never as pseudo-on-policy data:

    replay_loss = mean_over_buffer_sample( w_i * per_token_NLL(seq_i) )
    loss        = ppo_loss + replay_coef * replay_loss

## Two scale details that decide whether this collapses the run

1. **Per-token, not per-sequence.** A 40-character SMILES has sequence logP
   around -30 to -60, whereas the PPO surrogate is advantage-scaled, O(0.1-1).
   Summed-logP replay would outweigh the policy-gradient term by two orders
   of magnitude and simply supervised-train the policy onto the buffer --
   guaranteed collapse. Dividing by token count puts both terms at O(1) and
   makes `replay_coef` mean what it looks like it means.

2. **Weights normalised to mean 1.** Raw reward as a weight would re-create
   the v1 pathology inside the auxiliary term (one 0.95-reward molecule
   dominating a batch of 0.05s). Weights are reward-ranked, shifted positive
   and rescaled so the term's magnitude does not drift with the buffer's
   absolute reward level.

Even so this term is a mode-collapse *driver* by construction -- it is
maximum-likelihood on a fixed set of molecules with no clipping and no trust
region of its own. Three things bound it: `replay_coef` is small by default,
the buffer is scaffold-capped so the likelihood target can never become 69
copies of one core (which is precisely what v3's high-reward tail was), and
sampling is stratified over scaffolds rather than over molecules so a
populous bucket does not dominate a draw. The KL-to-prior controller in
train_rl.py also still sees the drift on fresh samples and ramps beta_kl, so
replay pulling the policy off-distribution is caught by the same mechanism
that caught v3's two runaway attempts.
"""
import random
import numpy as np
import torch


class ReplayBuffer:
    """Reward-ranked molecule memory with a hard per-scaffold cap.

    The cap is the whole point: an uncapped high-reward buffer on this
    project fills with the dominant cyclopropanecarboxamide-aminoquinoline
    series within a few hundred steps, and replaying it would actively
    reinforce the narrowness the diversity filter is trying to break.
    """

    def __init__(self, capacity=1000, max_per_scaffold=10, min_reward=0.0, seed=42):
        self.capacity = capacity
        self.max_per_scaffold = max_per_scaffold
        self.min_reward = min_reward
        self.by_scaffold = {}          # key -> list of (reward, smiles)
        self.seen = set()              # canonical SMILES already stored
        self.rng = random.Random(seed)
        self.n_offered = 0
        self.n_admitted = 0

    def __len__(self):
        return sum(len(v) for v in self.by_scaffold.values())

    def add(self, canon_smiles, keys, rewards):
        """Offer one batch. Admits novel, above-threshold molecules only.

        Duplicates are rejected outright -- unlike the diversity filter's
        counts (where repeat emissions must consume bucket capacity), the
        buffer is a likelihood *target*, and storing a molecule twice would
        double its pull for no additional information.
        """
        for s, k, r in zip(canon_smiles, keys, rewards):
            if not s or k is None:
                continue
            self.n_offered += 1
            if r < self.min_reward or s in self.seen:
                continue
            bucket = self.by_scaffold.setdefault(k, [])
            if len(bucket) >= self.max_per_scaffold:
                # Full bucket: replace its weakest member if this one beats it.
                lo = min(range(len(bucket)), key=lambda i: bucket[i][0])
                if bucket[lo][0] >= r:
                    continue
                self.seen.discard(bucket[lo][1])
                bucket[lo] = (float(r), s)
            else:
                bucket.append((float(r), s))
            self.seen.add(s)
            self.n_admitted += 1
        self._evict()

    def _evict(self):
        """Trim to capacity by dropping the globally weakest molecules.

        Evicting by reward alone would preferentially empty the exploratory
        low-reward scaffolds this buffer exists to preserve, so buckets of
        size 1 are protected -- a scaffold keeps at least one representative
        until its bucket is the only thing left to cut.
        """
        while len(self) > self.capacity:
            cands = [(r, k, i) for k, v in self.by_scaffold.items()
                     if len(v) > 1 for i, (r, _) in enumerate(v)]
            if not cands:
                cands = [(r, k, i) for k, v in self.by_scaffold.items()
                         for i, (r, _) in enumerate(v)]
            if not cands:
                return
            _, k, i = min(cands)
            self.seen.discard(self.by_scaffold[k][i][1])
            self.by_scaffold[k].pop(i)
            if not self.by_scaffold[k]:
                del self.by_scaffold[k]

    def sample(self, k):
        """Scaffold-stratified draw: pick scaffolds first, then a molecule.

        Sampling molecules uniformly would weight a scaffold by how many
        representatives it has, which is the concentration bias the cap
        already fights. Drawing scaffolds uniformly gives every chemotype in
        memory equal pull regardless of how easy it was to find.
        """
        if not self.by_scaffold:
            return [], np.zeros(0, dtype=np.float32)
        keys = list(self.by_scaffold)
        chosen = [self.rng.choice(keys) for _ in range(min(k, len(self)))]
        smiles, rewards = [], []
        for key in chosen:
            r, s = self.rng.choice(self.by_scaffold[key])
            smiles.append(s)
            rewards.append(r)
        w = np.asarray(rewards, dtype=np.float32)
        w = w - w.min() + 1e-3
        w = w / w.mean()
        return smiles, w

    def sample_diverse(self, n):
        """Up to `n` molecules, round-robin over scaffolds, best-first.

        Distinct from `sample()`: that draws with replacement for one RL
        update, this builds the fixed training set for a transfer-learning
        phase, where the same molecule appearing twice is wasted gradient and
        an over-represented scaffold is the failure mode being designed
        against. Round-robin takes each scaffold's best molecule before any
        scaffold's second, so truncating at `n` costs breadth last rather
        than first.
        """
        if not self.by_scaffold:
            return [], np.zeros(0, dtype=np.float32)
        ranked = {k: sorted(v, key=lambda t: -t[0]) for k, v in self.by_scaffold.items()}
        order = sorted(ranked, key=lambda k: -ranked[k][0][0])
        out = []
        depth = 0
        while len(out) < n:
            added = False
            for k in order:
                if depth < len(ranked[k]):
                    out.append(ranked[k][depth])
                    added = True
                    if len(out) >= n:
                        break
            if not added:
                break
            depth += 1
        rewards = np.asarray([r for r, _ in out], dtype=np.float32)
        w = rewards - rewards.min() + 1e-3
        return [s for _, s in out], (w / w.mean()).astype(np.float32)

    def stats(self):
        sizes = [len(v) for v in self.by_scaffold.values()]
        return {"size": len(self), "n_scaffolds": len(self.by_scaffold),
                "max_bucket": max(sizes) if sizes else 0,
                "n_admitted": self.n_admitted, "n_offered": self.n_offered}


def encode(smiles_list, vocab, max_len, device):
    """SMILES strings -> padded [BOS]...[EOS] id tensor.

    The vocabulary is character-level (preprocess_chembl.py mirrors OpenChem's
    `set(''.join(smiles))`, so 'Cl' is 'C' then 'l'), which makes the inverse
    of the sampler's decode a plain per-character lookup. Molecules containing
    a character outside the vocab, or longer than the sampler's own max_len,
    are dropped rather than truncated -- a truncated SMILES is not a molecule
    and training the policy to produce one is worse than replaying nothing.
    """
    PAD, BOS, EOS = vocab["pad"], vocab["bos"], vocab["eos"]
    rows, keep = [], []
    for i, s in enumerate(smiles_list):
        if len(s) + 2 > max_len or any(c not in vocab for c in s):
            continue
        rows.append([BOS] + [vocab[c] for c in s] + [EOS])
        keep.append(i)
    if not rows:
        return None, np.zeros(0, dtype=int)
    L = max(len(r) for r in rows)
    padded = torch.full((len(rows), L), PAD, dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        padded[i, :len(r)] = torch.tensor(r, device=device)
    return padded, np.asarray(keep, dtype=int)
