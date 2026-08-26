"""Resolve candidate protein names -> ChEMBL target IDs. Counts only, no datasets."""
import requests, time

BASE = "https://www.ebi.ac.uk/chembl/api/data"
S = requests.Session()

QUERIES = [
    ("ABL1",                    "ABL1",                    "Homo sapiens"),
    ("CDK2",                    "CDK2",                    "Homo sapiens"),
    ("Beta-2 adrenergic (?)",   "beta-2 adrenergic receptor", "Homo sapiens"),
    ("Androgen receptor (?)",   "androgen receptor",       "Homo sapiens"),
    ("A2A adenosine",           "adenosine A2a receptor",  "Homo sapiens"),
    ("Dopamine D2",             "dopamine D2 receptor",    "Homo sapiens"),
    ("AChE",                    "acetylcholinesterase",    "Homo sapiens"),
    ("HIV-1 protease",          "HIV-1 protease",          None),
    ("Carbonic anhydrase II",   "carbonic anhydrase II",   "Homo sapiens"),
]

for label, q, want_org in QUERIES:
    r = S.get(f"{BASE}/target/search.json", params={"q": q, "limit": 25}, timeout=60)
    hits = r.json().get("targets", [])
    single = [h for h in hits if h.get("target_type") == "SINGLE PROTEIN"]
    if want_org:
        single = [h for h in single if h.get("organism") == want_org] or single
    print(f"\n### {label}   (query={q!r}, organism={want_org})")
    if not single:
        print("   NO SINGLE PROTEIN HITS")
    for h in single[:4]:
        print(f"   {h['target_chembl_id']:<14} {h.get('organism','?'):<32} {h.get('pref_name','?')}")
    time.sleep(0.3)
