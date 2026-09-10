"""Pretrain a character-level SMILES RNN language model on ChEMBL.

Target-independent: this is the prior the RL stage will fine-tune. Faithful to
OpenChem's GenerativeRNN in the ways that matter (char-level vocab, LSTM stack,
teacher forcing, next-char cross-entropy) but written against modern PyTorch.

Length-bucketed batching keeps padding waste low; bf16 autocast for speed.
"""
import json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smiles_utils import canonicalize

from paths import DATA, CKPT
os.makedirs(CKPT, exist_ok=True)

EPOCHS      = int(os.environ.get("EPOCHS", 10))
BATCH       = 512
EMB, HID, LAYERS, DROPOUT = 128, 512, 3, 0.2
LR          = 1e-3
MAX_LEN     = 100
SEED        = 42
DEV         = torch.device("cuda")

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

vocab = json.load(open(f"{DATA}/vocab.json"))
itos = {i: s for s, i in vocab.items()}
PAD, BOS, EOS = vocab["pad"], vocab["bos"], vocab["eos"]
V = len(vocab)


def encode(smi):
    return np.array([BOS] + [vocab[c] for c in smi] + [EOS], dtype=np.uint8)


def load(split):
    with open(f"{DATA}/chembl_{split}.smi") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    return [encode(s) for s in lines], lines


print("loading corpus...", flush=True)
train_enc, _ = load("train")
val_enc, val_raw = load("val")
train_smiles_set = None   # filled lazily for novelty check
print(f"train {len(train_enc):,} | val {len(val_enc):,} | vocab {V}", flush=True)


def make_batches(encs, batch_size, shuffle=True):
    """Length-bucketed batches: sort by length, cut into batches, shuffle order."""
    idx = np.argsort([len(e) for e in encs], kind="stable")
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(encs, ids):
    seqs = [encs[i] for i in ids]
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.from_numpy(out)


class SmilesRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB, padding_idx=PAD)
        self.rnn = nn.LSTM(EMB, HID, LAYERS, batch_first=True,
                           dropout=DROPOUT if LAYERS > 1 else 0.0)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HID, V)

    def forward(self, x, h=None):
        e = self.emb(x)
        o, h = self.rnn(e, h)
        return self.fc(self.drop(o)), h


model = SmilesRNN().to(DEV)
nparam = sum(p.numel() for p in model.parameters())
print(f"model params: {nparam/1e6:.2f}M", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
lossf = nn.CrossEntropyLoss(ignore_index=PAD)


@torch.no_grad()
def sample(n=1000, temp=1.0, max_len=MAX_LEN):
    model.eval()
    x = torch.full((n, 1), BOS, dtype=torch.long, device=DEV)
    h = None
    done = torch.zeros(n, dtype=torch.bool, device=DEV)
    seqs = [[] for _ in range(n)]
    for _ in range(max_len):
        logits, h = model(x, h)
        probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
        probs[:, PAD] = 0; probs[:, BOS] = 0          # never emit pad/bos
        probs = probs / probs.sum(-1, keepdim=True)
        nxt = torch.multinomial(probs, 1)
        for i, t in enumerate(nxt.squeeze(1).tolist()):
            if not done[i] and t != EOS:
                seqs[i].append(itos[t])
        done |= (nxt.squeeze(1) == EOS)
        if done.all():
            break
        x = nxt
    return ["".join(s) for s in seqs]


@torch.no_grad()
def evaluate():
    model.eval()
    tot, ntok = 0.0, 0
    for ids in make_batches(val_enc, BATCH, shuffle=False):
        b = collate(val_enc, ids).to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b[:, :-1])
            loss = lossf(logits.reshape(-1, V).float(), b[:, 1:].reshape(-1))
        n = (b[:, 1:] != PAD).sum().item()
        tot += loss.item() * n; ntok += n
    return tot / ntok


print(f"\n{'epoch':>5}{'train_loss':>12}{'val_loss':>10}{'ppl':>8}"
      f"{'valid%':>9}{'unique%':>9}{'mins':>7}", flush=True)
print("-" * 62, flush=True)

best = float("inf")
history = []
for ep in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    batches = make_batches(train_enc, BATCH)
    run, seen = 0.0, 0
    for step, ids in enumerate(batches):
        b = collate(train_enc, ids).to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b[:, :-1])
            loss = lossf(logits.reshape(-1, V).float(), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        run += loss.item(); seen += 1
        if step % 500 == 0:
            print(f"    ep{ep} step {step}/{len(batches)} loss {run/seen:.4f}", flush=True)
    sched.step()

    tr = run / seen
    vl = evaluate()
    gen = sample(1000)
    canon = [canonicalize(s) for s in gen]
    ok = [c for c in canon if c]
    validp = 100 * len(ok) / len(gen)
    uniqp = 100 * len(set(ok)) / max(1, len(ok))
    mins = (time.time() - t0) / 60
    print(f"{ep:>5}{tr:>12.4f}{vl:>10.4f}{math.exp(vl):>8.3f}"
          f"{validp:>9.1f}{uniqp:>9.1f}{mins:>7.1f}", flush=True)
    history.append({"epoch": ep, "train_loss": tr, "val_loss": vl,
                    "valid_pct": validp, "unique_pct": uniqp})

    if vl < best:
        best = vl
        torch.save({"model": model.state_dict(), "vocab": vocab, "epoch": ep,
                    "val_loss": vl, "config": {"emb": EMB, "hid": HID,
                    "layers": LAYERS, "dropout": DROPOUT, "max_len": MAX_LEN}},
                   f"{CKPT}/generator_best.pt")
        print(f"      saved new best (val {vl:.4f})", flush=True)

json.dump(history, open(f"{CKPT}/pretrain_history.json", "w"), indent=1)

# ---- final report against the held-out set
print("\n=== FINAL SAMPLE QUALITY (best checkpoint) ===", flush=True)
ck = torch.load(f"{CKPT}/generator_best.pt", weights_only=False)
model.load_state_dict(ck["model"])
gen = sample(5000)
canon = [canonicalize(s) for s in gen]
ok = [c for c in canon if c]
train_set = set(open(f"{DATA}/chembl_train.smi").read().split())
novel = [c for c in set(ok) if c not in train_set]
print(f"sampled          : {len(gen)}")
print(f"valid            : {len(ok)} ({100*len(ok)/len(gen):.1f}%)")
print(f"unique (of valid): {len(set(ok))} ({100*len(set(ok))/max(1,len(ok)):.1f}%)")
print(f"novel (of unique): {len(novel)} ({100*len(novel)/max(1,len(set(ok))):.1f}%)")
print(f"best val loss    : {ck['val_loss']:.4f} (epoch {ck['epoch']})")
print("examples:")
for s in ok[:10]:
    print("   ", s)
print("PRETRAIN_DONE")
