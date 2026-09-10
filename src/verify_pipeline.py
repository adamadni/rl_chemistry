"""End-to-end smoke test: ChEMBL corpus -> RDKit validity -> OpenChem tokenizer.

Verifies the plumbing only. No training, no model construction.
"""
import gzip
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RAW
from smiles_utils import canonicalize, is_valid, valid_fraction, largest_fragment

CHEMREPS = os.path.join(RAW, "chembl_37_chemreps.txt.gz")
N_SAMPLE = 10000
TOTAL_ROWS = 2897819


def sample_smiles(n=N_SAMPLE):
    """Strided sample so we span the whole file, not just the peptide-heavy head."""
    stride = max(1, TOTAL_ROWS // n)
    out = []
    with gzip.open(CHEMREPS, "rt") as fh:
        next(fh)  # header
        for i, line in enumerate(fh):
            if i % stride:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                out.append(parts[1])
            if len(out) >= n:
                break
    return out


def hr(title):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


hr("1. SAMPLE ChEMBL 37")
t0 = time.time()
smiles = sample_smiles()
print(f"sampled {len(smiles)} SMILES (strided across {TOTAL_ROWS} rows) in {time.time()-t0:.1f}s")
print(f"example: {smiles[0][:70]}")

hr("2. RDKit VALIDITY on real ChEMBL data")
t0 = time.time()
frac = valid_fraction(smiles)
dt = time.time() - t0
print(f"valid fraction : {frac:.4f}  ({int(frac*len(smiles))}/{len(smiles)})")
print(f"throughput     : {len(smiles)/dt:,.0f} SMILES/s  ({dt:.1f}s)")
print("expectation    : ~1.00, ChEMBL SMILES are already RDKit-canonical")

hr("3. NEGATIVE CONTROL (these MUST be rejected)")
bad = {
    "unclosed ring":      "C1CC",
    "unclosed aromatic":  "c1ccccc",
    "5-valent nitrogen":  "N(C)(C)(C)(C)C",
    "nonsense atoms":     "XYZ",
    "empty string":       "",
    "dangling paren":     "C(",
    "divalent fluorine":  "F(C)C",
    "malformed salt":     "[Cl-].[Na+",
}
# NB: 'c1ccc1' is deliberately NOT here -- RDKit parses it as cyclobutadiene
# (C1=CC=C1), de-aromatizing the 4-ring. That is correct behaviour, not a leak.
n_rej = 0
for label, sm in bad.items():
    v = is_valid(sm)
    n_rej += not v
    print(f"  {'REJECTED' if not v else 'ACCEPTED <-- PROBLEM'}  {label:<20} {sm!r}")
print(f"\nrejected {n_rej}/{len(bad)} invalid inputs")

hr("4. CANONICALIZATION ROUND-TRIP (idempotence)")
# Same molecule written 3 ways must collapse to one canonical string.
variants = ["C1=CC=CC=C1", "c1ccccc1", "C1=CC=CC=C1"]
canon = [canonicalize(s) for s in variants]
print(f"benzene variants -> {canon}")
print(f"all identical    : {len(set(canon)) == 1}")

sub = smiles[:2000]
once = [canonicalize(s) for s in sub]
twice = [canonicalize(s) for s in once]
stable = sum(a == b for a, b in zip(once, twice))
print(f"idempotent on {stable}/{len(sub)} ChEMBL molecules")

hr("5. SALT STRIPPING")
for sm in ["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "Cl.CN1CCC[C@H]1c1cccnc1"]:
    print(f"  {sm}\n    -> {largest_fragment(sm)}")

hr("6. OpenChem TOKENIZER on the same batch")
from openchem.data.utils import get_tokens, sanitize_smiles

tokens, token2idx, num_tokens = get_tokens(sub)
print(f"vocabulary size : {num_tokens}")
print(f"alphabet        : {''.join(sorted(t for t in tokens if t.strip()))}")

# sanitize_smiles returns (new_smiles, idx); idx = positions that passed.
_, idx = sanitize_smiles(sub, logging="none")
_, idx_c = sanitize_smiles(sub, allow_charges=True, logging="none")
print(f"kept {len(idx)}/{len(sub)}  (default allow_charges=False)")
print(f"kept {len(idx_c)}/{len(sub)}  (allow_charges=True)")
print(f"-> charge filter alone drops {len(idx_c) - len(idx)} molecules")

# OpenChem calls MolFromSmiles(sanitize=False): syntax errors are caught,
# valence errors are NOT. Our strict checker differs -- quantify where.
print("\nOpenChem sanitize_smiles vs strict RDKit on the invalid set:")
for label, sm in bad.items():
    _, i2 = sanitize_smiles([sm], logging="none")
    oc = "accept" if len(i2) else "reject"
    rd = "accept" if is_valid(sm) else "reject"
    flag = "   <-- DISAGREE" if oc != rd else ""
    print(f"  {label:<20} openchem={oc:<7} rdkit={rd}{flag}")

hr("7. LENGTH DISTRIBUTION (informs max_len for pretraining)")
lens = sorted(len(s) for s in smiles)
n = len(lens)
for p in (50, 75, 90, 95, 99, 100):
    print(f"  p{p:<3} = {lens[min(n-1, p*n//100)]:>5} chars")
print(f"  mean = {sum(lens)/n:.1f} chars")
print(f"  <=100 chars: {100*sum(l<=100 for l in lens)/n:.1f}%   <=120: {100*sum(l<=120 for l in lens)/n:.1f}%")

print("\n" + "=" * 62)
print("PIPELINE VERIFICATION COMPLETE")
print("=" * 62)
