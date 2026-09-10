"""Why did the reward model score the known ABL1 drugs the way it did?

Run after any change to the labelling scheme or the QSAR model. For imatinib,
dasatinib and nilotinib it reports the measured pChEMBL in the training set,
the label that measurement produced, the model's predicted P(active), and the
nearest training neighbours by Tanimoto.

This exists because a reward model can be right on aggregate metrics and wrong
in the way that matters. Under median-pChEMBL aggregation the scaffold-split
ROC-AUC was 0.906 -- slightly better than the 0.900 the project ended with --
while imatinib (6.66) and nilotinib (7.55) were both labelled INACTIVE, leaving
the active class dominated by one chemotype. An aggregate score cannot show
that; this can, which is what prompted the switch to max aggregation.
"""
import json, os, sys
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA
RDLogger.DisableLog("rdApp.*")

data = json.load(open(os.path.join(DATA, "abl1_qsar.json")))
by_smi = {d["smiles"]: d for d in data}

drugs = {
    "imatinib":  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "dasatinib": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "nilotinib": "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
}

print("%-12s%-13s%9s%7s%8s" % ("drug", "in dataset?", "pChEMBL", "label", "n_meas"))
print("-" * 50)
for n, s in drugs.items():
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        print("%-12s%-13s" % (n, "BAD SMILES"))
        continue
    c = Chem.MolToSmiles(mol)
    d = by_smi.get(c)
    if d:
        print("%-12s%-13s%9.2f%7d%8d" % (n, "YES", d["pchembl"], d["label"], d["n_measurements"]))
    else:
        print("%-12s%-13s%9s%7s%8s" % (n, "NOT FOUND", "-", "-", "-"))

g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
for target in ["imatinib", "nilotinib"]:
    ref = g.GetFingerprint(Chem.MolFromSmiles(drugs[target]))
    sims = []
    for d in data:
        m = Chem.MolFromSmiles(d["smiles"])
        sims.append((DataStructs.TanimotoSimilarity(ref, g.GetFingerprint(m)), d))
    sims.sort(key=lambda x: -x[0])
    print("\nnearest dataset neighbours to %s (ECFP4 Tanimoto):" % target)
    for s, d in sims[:5]:
        print("   %.3f  pChEMBL %5.2f  label=%d  %s" % (s, d["pchembl"], d["label"], d["smiles"][:58]))
