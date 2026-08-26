"""Screen candidate ChEMBL targets: counts only, no dataset downloads.

Every number here is a `page_meta.total_count` from a limit=1 query.
Confidence distribution comes from the ASSAY endpoint -- the activity
endpoint has no confidence_score field and silently ignores the filter.
"""
import requests, time, sys

BASE = "https://www.ebi.ac.uk/chembl/api/data"
S = requests.Session()
SLEEP = 0.15


def count(endpoint, **params):
    params.update(limit=1, format="json")
    for attempt in range(4):
        try:
            r = S.get(f"{BASE}/{endpoint}.json", params=params, timeout=90)
            if r.status_code == 200:
                time.sleep(SLEEP)
                return r.json()["page_meta"]["total_count"]
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


TARGETS = [
    ("ABL1",                  "CHEMBL1862"),
    ("CDK2",                  "CHEMBL301"),
    ("Beta-2 adrenergic R",   "CHEMBL210"),
    ("Androgen receptor",     "CHEMBL1871"),
    ("A2A adenosine R",       "CHEMBL251"),
    ("Dopamine D2 R",         "CHEMBL217"),
    ("AChE",                  "CHEMBL220"),
    ("HIV-1 protease",        "CHEMBL243"),
    ("Carbonic anhydrase 2",  "CHEMBL205"),
]

# ---------------------------------------------------------------- sanity
print("=" * 78)
print("FILTER VALIDATION (the API silently ignores unknown filters, so prove")
print("each filter actually changes the count before trusting any number)")
print("=" * 78)
ref = "CHEMBL301"
n_all  = count("activity", target_chembl_id=ref)
n_pchembl = count("activity", target_chembl_id=ref, pchembl_value__isnull="false")
n_ge0  = count("activity", target_chembl_id=ref, pchembl_value__gte=0)
n_ge99 = count("activity", target_chembl_id=ref, pchembl_value__gte=99)
print(f"  all activities              = {n_all}")
print(f"  pchembl_value__isnull=false = {n_pchembl}   (must be < all)          {'PASS' if n_pchembl < n_all else 'FAIL'}")
print(f"  pchembl_value__gte=0        = {n_ge0}   (must equal isnull=false) {'PASS' if n_ge0 == n_pchembl else 'FAIL'}")
print(f"  pchembl_value__gte=99       = {n_ge99}      (must be 0)               {'PASS' if n_ge99 == 0 else 'FAIL'}")
if not (n_pchembl < n_all and n_ge0 == n_pchembl and n_ge99 == 0):
    sys.exit("filter validation FAILED -- numbers would be meaningless")

# ---------------------------------------------------------------- identity
print("\n" + "=" * 78)
print("TARGET IDENTITY CHECK")
print("=" * 78)
meta = {}
for label, tid in TARGETS:
    r = S.get(f"{BASE}/target/{tid}.json", timeout=60).json()
    meta[tid] = r
    print(f"  {label:<22} {tid:<12} {r.get('target_type','?'):<15} "
          f"{r.get('organism','?'):<32} {r.get('pref_name','?')}")
    time.sleep(SLEEP)

# ---------------------------------------------------------------- screen
rows = []
for label, tid in TARGETS:
    d = {"label": label, "tid": tid}
    d["all"]     = count("activity", target_chembl_id=tid)
    d["pchembl"] = count("activity", target_chembl_id=tid, pchembl_value__isnull="false")
    d["ge6"]     = count("activity", target_chembl_id=tid, pchembl_value__gte=6)
    d["ge7"]     = count("activity", target_chembl_id=tid, pchembl_value__gte=7)
    d["ge8"]     = count("activity", target_chembl_id=tid, pchembl_value__gte=8)
    d["assays"]  = count("assay", target_chembl_id=tid)
    d["conf"]    = {c: count("assay", target_chembl_id=tid, confidence_score=c) for c in range(10)}
    rows.append(d)
    print(f"  ...screened {label}", flush=True)

# ---------------------------------------------------------------- report
print("\n" + "=" * 78)
print("BIOACTIVITY COUNTS")
print("=" * 78)
print(f"{'target':<22}{'ChEMBL ID':<13}{'all':>9}{'pChEMBL':>9}{'>=6':>8}{'>=7':>8}{'>=8':>8}")
print("-" * 78)
for d in rows:
    print(f"{d['label']:<22}{d['tid']:<13}{d['all']:>9,}{d['pchembl']:>9,}"
          f"{d['ge6']:>8,}{d['ge7']:>8,}{d['ge8']:>8,}")

print("\n" + "=" * 78)
print("ASSAY CONFIDENCE-SCORE DISTRIBUTION  (9=direct single protein, 8=homologous)")
print("=" * 78)
print(f"{'target':<22}{'assays':>8}" + "".join(f"{('c'+str(c)):>7}" for c in range(4, 10)) + f"{'%>=8':>8}")
print("-" * 78)
for d in rows:
    hi = d["conf"][8] + d["conf"][9]
    pct = 100 * hi / d["assays"] if d["assays"] else 0
    low = sum(d["conf"][c] for c in range(0, 4))
    print(f"{d['label']:<22}{d['assays']:>8,}" +
          "".join(f"{d['conf'][c]:>7,}" for c in range(4, 10)) + f"{pct:>7.1f}%")

print("\n" + "=" * 78)
print("SPARSITY VERDICT  (threshold: ~500 actives at pChEMBL >= 8)")
print("=" * 78)
for d in sorted(rows, key=lambda x: -x["ge8"]):
    verdict = "TOO SPARSE" if d["ge8"] < 500 else ("marginal" if d["ge8"] < 1000 else "ample")
    print(f"  {d['label']:<22} {d['ge8']:>7,} actives   {verdict}")
