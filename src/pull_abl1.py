"""Pull ABL1 (CHEMBL1862) bioactivity and build the QSAR dataset.
One row per unique compound. Where a compound has several measurements we take
the MAX pChEMBL (a compound counts active if *any* assay showed top-decile
potency) and label active at pChEMBL >= 8, matching the screening threshold.

Switched from median -> max aggregation (2026-08-24): median put known ABL1
drugs (imatinib 6.66, nilotinib 7.55) in the inactive class and left the
active class dominated by dasatinib-like chemistry, giving the RL reward
model a scaffold-narrow target. Max aggregation is more outlier-sensitive
(one noisy high-potency replicate can flip a label) but pulls chemically
diverse known actives into the positive class, broadening what the reward
model treats as "active" -- see CLAUDE.md decision #1.
"""
import json, time, os
from collections import defaultdict
import requests
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET = "CHEMBL1862"
OUT = "/workspace/rl_chemistry/data/processed"
os.makedirs(OUT, exist_ok=True)
ACTIVE_THRESHOLD = 8.0
# binding/functional potency only -- excludes ADMET/tox endpoints
KEEP_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}
S = requests.Session()
rows, offset, LIMIT = [], 0, 1000
while True:
    for attempt in range(5):
        try:
            r = S.get(f"{BASE}/activity.json", timeout=120, params={
                "target_chembl_id": TARGET,
                "pchembl_value__isnull": "false",
                "limit": LIMIT, "offset": offset,
                "only": "molecule_chembl_id,canonical_smiles,pchembl_value,"
                        "standard_type,assay_chembl_id,data_validity_comment",
            })
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit(f"API failed at offset {offset}")
    d = r.json()
    batch = d["activities"]
    rows.extend(batch)
    total = d["page_meta"]["total_count"]
    offset += LIMIT
    print(f"  pulled {len(rows)}/{total}", flush=True)
    if offset >= total or not batch:
        break
    time.sleep(0.2)
print(f"\nraw activity records: {len(rows):,}")
type_counts = defaultdict(int)
for r_ in rows:
    type_counts[r_.get("standard_type")] += 1
print("standard_type distribution (top 10):")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"   {str(t):<12} {c:>7,}   {'KEEP' if t in KEEP_TYPES else 'drop'}")
# ---- filter + aggregate per compound
flagged = sum(1 for r_ in rows if r_.get("data_validity_comment"))
print(f"\ndropped {flagged:,} records with a data_validity_comment (suspect data)")
per_cmpd = defaultdict(list)
smiles_of = {}
kept_records = 0
for r_ in rows:
    if r_.get("data_validity_comment"):
        continue
    if r_.get("standard_type") not in KEEP_TYPES:
        continue
    smi, cid, pv = r_.get("canonical_smiles"), r_.get("molecule_chembl_id"), r_.get("pchembl_value")
    if not (smi and cid and pv):
        continue
    per_cmpd[cid].append(float(pv))
    smiles_of[cid] = smi
    kept_records += 1
print(f"records kept after type/validity filter: {kept_records:,}")
print(f"unique compounds: {len(per_cmpd):,}")
def clean(smi):
    """Salt-strip + canonicalize, same convention as the pretraining corpus."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))
def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
data, bad_smiles, disagree = [], 0, 0
for cid, vals in per_cmpd.items():
    smi = clean(smiles_of[cid])
    if smi is None:
        bad_smiles += 1
        continue
    # flag compounds whose replicates straddle the activity threshold
    if len(vals) > 1 and min(vals) < ACTIVE_THRESHOLD <= max(vals):
        disagree += 1
    p = max(vals)
    data.append({"chembl_id": cid, "smiles": smi, "pchembl": p,
                 "pchembl_median": median(vals),
                 "n_measurements": len(vals),
                 "label": int(p >= ACTIVE_THRESHOLD)})
print(f"dropped {bad_smiles} compounds with unparseable SMILES")
print(f"note: {disagree:,} compounds have replicates straddling pChEMBL {ACTIVE_THRESHOLD} "
      f"(max assignment resolves them; pchembl_median kept for reference)")
# dedup identical canonical SMILES arriving from different ChEMBL IDs
by_smiles = {}
for d_ in data:
    prev = by_smiles.get(d_["smiles"])
    if prev is None or d_["n_measurements"] > prev["n_measurements"]:
        by_smiles[d_["smiles"]] = d_
collapsed = len(data) - len(by_smiles)
data = list(by_smiles.values())
print(f"collapsed {collapsed} duplicate canonical structures")
n_act = sum(d_["label"] for d_ in data)
print(f"\nFINAL DATASET: {len(data):,} compounds")
print(f"  actives  (max pChEMBL >= {ACTIVE_THRESHOLD}): {n_act:,}  ({100*n_act/len(data):.1f}%)")
print(f"  inactives:                     {len(data)-n_act:,}")
with open(f"{OUT}/abl1_qsar.json", "w") as fh:
    json.dump(data, fh)
print(f"wrote {OUT}/abl1_qsar.json")
