"""Build the pretraining corpus from ChEMBL 37 chemreps.

Pipeline per molecule: largest fragment (salt strip) -> canonicalize ->
element whitelist -> size/length filters -> dedup. Character-level vocabulary,
matching OpenChem's `get_tokens` (it does set(''.join(smiles)), so 'Cl' is
two tokens 'C','l'). Staying faithful to the reference here.
"""
import gzip, json, os, random, re, sys
from multiprocessing import Pool
from collections import Counter

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RAW as RAW_DIR, DATA as OUT

RAW = os.path.join(RAW_DIR, "chembl_37_chemreps.txt.gz")
os.makedirs(OUT, exist_ok=True)

MAX_LEN = 100          # chars; covers ~95% of ChEMBL (p95=102 measured earlier)
MIN_HEAVY, MAX_HEAVY = 10, 50
ALLOWED = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"}
VAL_FRAC = 0.05
SEED = 42

# reason codes
OK, E_PARSE, E_FRAG, E_ELEM, E_SIZE, E_LEN = range(6)


def process(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return E_PARSE, None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return E_FRAG, None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())   # drop counterions
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ALLOWED:
            return E_ELEM, None
    n = mol.GetNumHeavyAtoms()
    if not (MIN_HEAVY <= n <= MAX_HEAVY):
        return E_SIZE, None
    out = Chem.MolToSmiles(mol)          # canonical, stereo preserved
    if len(out) > MAX_LEN:
        return E_LEN, None
    return OK, out


def read_raw():
    with gzip.open(RAW, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                yield p[1]


if __name__ == "__main__":
    print("reading + filtering (32 procs)...", flush=True)
    counts = Counter()
    kept = []
    with Pool(30) as pool:
        for code, smi in pool.imap_unordered(process, read_raw(), chunksize=2000):
            counts[code] += 1
            if code == OK:
                kept.append(smi)
    total = sum(counts.values())

    print(f"\n{'stage':<28}{'count':>12}{'pct':>9}")
    print("-" * 49)
    for code, name in [(E_PARSE, "dropped: unparseable"), (E_FRAG, "dropped: no fragment"),
                       (E_ELEM, "dropped: element not allowed"), (E_SIZE, f"dropped: heavy atoms outside {MIN_HEAVY}-{MAX_HEAVY}"),
                       (E_LEN, f"dropped: len > {MAX_LEN}"), (OK, "passed filters")]:
        print(f"{name:<28}{counts[code]:>12,}{100*counts[code]/total:>8.2f}%")
    print(f"{'input total':<28}{total:>12,}")

    before = len(kept)
    kept = list(dict.fromkeys(kept))     # dedup, order-stable
    print(f"\ndedup: {before:,} -> {len(kept):,}  (removed {before-len(kept):,} duplicates)")

    random.seed(SEED)
    random.shuffle(kept)
    n_val = int(len(kept) * VAL_FRAC)
    val, train = kept[:n_val], kept[n_val:]

    for name, data in [("train", train), ("val", val)]:
        with open(f"{OUT}/chembl_{name}.smi", "w") as fh:
            fh.write("\n".join(data) + "\n")
        print(f"wrote {name:<5} {len(data):>9,} -> {OUT}/chembl_{name}.smi")

    chars = sorted(set("".join(kept)))
    vocab = {"pad": 0, "bos": 1, "eos": 2}
    for c in chars:
        vocab[c] = len(vocab)
    json.dump(vocab, open(f"{OUT}/vocab.json", "w"), indent=1)
    print(f"\nvocab: {len(vocab)} tokens (3 special + {len(chars)} chars)")
    print("alphabet:", "".join(chars))

    lens = [len(s) for s in kept]
    stats = {"n_total_input": total, "n_kept": len(kept), "n_train": len(train),
             "n_val": len(val), "vocab_size": len(vocab), "max_len": MAX_LEN,
             "mean_len": sum(lens)/len(lens), "max_observed_len": max(lens)}
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
    print(f"mean SMILES length: {stats['mean_len']:.1f} chars")
