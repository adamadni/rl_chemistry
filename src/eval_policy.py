"""Compare an RL-finetuned policy against the pretrained prior.

Samples N molecules from each and reports the metrics ENGINEERING_LOG.md commits to
(valid / unique / novel fractions, predicted-active fraction) plus the
mode-collapse and applicability-domain diagnostics used during training.
Scaffold diversity is the headline number here: the whole point of the
threshold-shaping + PPO + adaptive-KL work was to raise reward WITHOUT
collapsing scaffold variety, so both must be read together.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smiles_utils import canonicalize
from reward_model import RewardModel
from train_rl import SmilesRNN, ADReference, scaffold_of, DATA, CKPT, DEV
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
@torch.no_grad()
def sample(model, vocab, n, max_len, temp=1.0, chunk=512):
    PAD, BOS, EOS = vocab["pad"], vocab["bos"], vocab["eos"]
    itos = {i: s for s, i in vocab.items() if s not in ("pad", "bos", "eos")}
    out = []
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        x = torch.full((b, 1), BOS, dtype=torch.long, device=DEV)
        h = None
        done = torch.zeros(b, dtype=torch.bool, device=DEV)
        toks = [[] for _ in range(b)]
        for _ in range(max_len):
            logits, h = model(x, h)
            logits = logits[:, -1, :].clone() / temp
            logits[:, PAD] = -1e9
            logits[:, BOS] = -1e9
            nxt = torch.distributions.Categorical(logits=logits).sample()
            for i, t in enumerate(nxt.tolist()):
                if not done[i] and t != EOS:
                    toks[i].append(t)
            done |= (nxt == EOS)
            if done.all():
                break
            x = nxt.unsqueeze(1)
        out.extend("".join(itos[t] for t in ids if t in itos) for ids in toks)
    return out
def load(path):
    ck = torch.load(path, weights_only=False)
    cfg, vocab = ck["config"], ck["vocab"]
    m = SmilesRNN(len(vocab), cfg["emb"], cfg["hid"], cfg["layers"], dropout=0.0).to(DEV)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, vocab, ck.get("step")
def report(name, smiles, rm, ad_ref, train_set):
    canon = [canonicalize(s) if s else None for s in smiles]
    valid = [c for c in canon if c]
    uniq = sorted(set(valid))
    novel = [c for c in uniq if c not in train_set]
    score_in = [c if c else "" for c in canon]
    sc = rm.score(score_in)
    vmask = np.array([c is not None for c in canon])
    # reward-model view, valid molecules only
    p_valid = sc["p_active"][vmask]
    r_valid = sc["reward"][vmask]
    u_valid = sc["uncertainty"][vmask]
    scafs = {s for s in (scaffold_of(c) for c in uniq) if s is not None}
    max_tan = ad_ref.max_tanimoto(score_in, vmask)
    order = np.argsort(-sc["reward"])
    topd = order[:max(1, len(order) // 10)]
    print(f"\n=== {name} (n={len(smiles)}) ===")
    print(f"  valid                {100*len(valid)/len(smiles):.1f}%")
    print(f"  unique (of valid)    {100*len(uniq)/max(1,len(valid)):.1f}%")
    print(f"  novel  (of unique)   {100*len(novel)/max(1,len(uniq)):.1f}%   (vs ChEMBL train)")
    print(f"  unique scaffolds     {len(scafs)}  ({len(scafs)/max(1,len(uniq)):.3f} per unique mol)")
    print(f"  mean P(active)       {p_valid.mean():.4f}")
    print(f"  mean reward          {r_valid.mean():.4f}   (P(active) - uncertainty)")
    print(f"  mean RF uncertainty  {u_valid.mean():.4f}")
    print(f"  P(active) >= 0.5     {100*(p_valid >= 0.5).mean():.2f}%  of valid")
    print(f"  P(active) >= 0.8     {100*(p_valid >= 0.8).mean():.2f}%  of valid")
    print(f"  AD: mean max-Tanimoto to train   {max_tan[vmask].mean():.3f}")
    print(f"  AD: top-reward decile within 0.3 {100*(max_tan[topd] >= 0.3).mean():.1f}%")
    return {"valid_pct": 100*len(valid)/len(smiles),
            "unique_pct": 100*len(uniq)/max(1,len(valid)),
            "novel_pct": 100*len(novel)/max(1,len(uniq)),
            "n_scaffolds": len(scafs),
            "scaffold_ratio": len(scafs)/max(1,len(uniq)),
            "mean_p_active": float(p_valid.mean()),
            "mean_reward": float(r_valid.mean()),
            "mean_uncertainty": float(u_valid.mean()),
            "pct_p_ge_0.5": float(100*(p_valid >= 0.5).mean()),
            "pct_p_ge_0.8": float(100*(p_valid >= 0.8).mean()),
            "ad_mean_max_tanimoto": float(max_tan[vmask].mean()),
            "ad_pct_top_decile": float(100*(max_tan[topd] >= 0.3).mean())}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=f"{CKPT}/rl_baseline_v3/policy_latest.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default=f"{CKPT}/rl_baseline_v3/eval.json")
    args = ap.parse_args()
    rm = RewardModel()
    ad_ref = ADReference()
    train_set = set(open(f"{DATA}/chembl_train.smi").read().split())
    prior, vocab_p, _ = load(f"{CKPT}/generator_best.pt")
    policy, vocab, step = load(args.policy)
    print(f"policy: {args.policy} (step {step})")
    res = {}
    res["prior"] = report("PRETRAINED PRIOR", sample(prior, vocab_p, args.n, args.max_len, args.temp),
                          rm, ad_ref, train_set)
    res["policy"] = report(f"RL POLICY (step {step})",
                           sample(policy, vocab, args.n, args.max_len, args.temp),
                           rm, ad_ref, train_set)
    print("\n=== DELTA (policy - prior) ===")
    for k in res["prior"]:
        d = res["policy"][k] - res["prior"][k]
        print(f"  {k:<28} {res['prior'][k]:>9.3f} -> {res['policy'][k]:>9.3f}  ({d:+.3f})")
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
