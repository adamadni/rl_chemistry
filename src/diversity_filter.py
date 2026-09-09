"""Scaffold-occupancy memory that decays reward on over-mined chemotypes.

This is the direct fix for v3's headline limitation. v3 does not collapse
globally (96% unique, 2938 scaffolds over the whole sample) but its
high-reward tail is 118 molecules over 33 Bemis-Murcko scaffolds with one
covering 58% -- R-group enumeration around a single core, not scaffold
hopping. Nothing in v3's objective (`P(active) - lambda*unc`, trust-region
bounded by KL to the prior) contains any term that cares *which* scaffold a
molecule has, so PPO and the KL controller structurally cannot fix this:
they bound how fast the policy climbs the reward surface, not which hill it
climbs. The reward surface itself has to change, and that is what this does.

Mechanism (Blaschke et al. 2020, J Cheminform 12:68, "memory-assisted RL"):
keep a count per scaffold of the high-reward molecules already produced, and
decay the reward of a molecule whose scaffold bucket is filling up, so
continuing to mine a solved chemotype stops paying and the only remaining
source of advantage is somewhere else.

## Why the scaffold key is the generic framework, not the Murcko scaffold

Bemis-Murcko keeps ring identity, so on v3's own output these are three
*different* scaffolds:

    O=C(Nc1cc2cc(-c3ccccc3)ccc2cn1)C1CC1     69 molecules
    O=C(Nc1cc2cc(-c3ccccn3)ccc2cn1)C1CC1      8
    O=C(Nc1cc2cc(-c3cccnc3)ccc2cn1)C1CC1      6

They are the same chemotype with phenyl / 2-pyridyl / 3-pyridyl hung off one
position. A filter keyed on Murcko would be trivially evadable: swap one ring
nitrogen, land in a fresh bucket, keep enumerating. It would also make the
headline metric look like it improved while nothing chemically changed --
v3's "33 scaffolds" is itself inflated for exactly this reason.

`MakeScaffoldGeneric` maps every atom to carbon and every bond to single, so
all three collapse to one framework and the filter bites on the actual core
topology. The cost is that it is blind to heteroatom placement, which can
merge two genuinely distinct chemotypes that share a ring skeleton; that is
the conservative direction of error here (it under-counts diversity rather
than over-counting it), and both keys are tracked in diagnostics so the
difference stays visible.

## Why decay-to-zero rather than REINVENT's score->0

REINVENT zeroes the score of a filtered molecule because its score is in
[0, 1] and 0 is the floor. Here reward is deliberately NOT floor-clipped
(see reward_model.py: clipping to 0 is what tied the early reward landscape
and caused the v1 collapse), so a prior-quality sample sits near -0.17 and an
invalid one at -1.0. Assigning 0.0 to a filtered molecule would therefore
*promote* it above ordinary chemistry. Instead the multiplier applies only to
the positive part: a fully-saturated scaffold earns exactly 0.0 -- better
than junk, worse than any unexplored molecule that scores at all -- and once
ThresholdShaper's tau is above 0 the shaped reward `0 - tau` is actively
negative, so the policy is pushed off the mined core rather than merely
un-rewarded for it.

Only molecules at or above `record_threshold` P(active) are counted. Filling
buckets with everything the policy emits would penalize scaffolds it never
actually exploited, which suppresses exploration instead of redirecting it.
"""
import collections
import functools
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# The policy emits the same molecule many times per run (that is the behaviour
# being measured), and both the filter and the replay buffer key on every
# sampled molecule every step, so memoising this is worth more than it looks:
# it turns four RDKit calls per duplicate into a dict hit.
@functools.lru_cache(maxsize=200_000)
def scaffold_key(smiles, mode="generic"):
    """Bucket identity for `smiles`, or None if RDKit cannot produce one.

    mode="murcko"  -- Bemis-Murcko scaffold (ring systems + linkers, atom
                      and bond types preserved)
    mode="generic" -- the same skeleton with all atoms -> C and all bonds ->
                      single, so ring-heteroatom swaps share a bucket
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        if mode == "murcko":
            return Chem.MolToSmiles(core)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        # MakeScaffoldGeneric rewrites atoms to carbon and bonds to single
        # without re-perceiving the result, so an aromatic ring can come back
        # flagged aromatic with single bonds. Canonical SMILES of that mol is
        # not stable, which would silently split one framework across several
        # buckets -- exactly the failure this key exists to prevent. Sanitize
        # so the ring perception is redone before canonicalisation.
        Chem.SanitizeMol(generic)
        return Chem.MolToSmiles(generic)
    except Exception:
        # MakeScaffoldGeneric and SanitizeMol both raise on a few exotic
        # valences/charges; an unbucketable molecule is left unfiltered
        # rather than dropped.
        return None


class DiversityFilter:
    """Per-scaffold occupancy counter and the reward multiplier it implies.

    multiplier(key) = clamp(1 - count[key] / bucket_size, 0, 1)

    Linear rather than a hard cliff at bucket_size: a cliff makes reward
    discontinuous in a way the EMA baseline lags badly (the batch mean drops
    the step a popular bucket fills, so every *other* molecule in that batch
    gets a spurious positive advantage). The ramp spreads that over
    bucket_size molecules instead.
    """

    def __init__(self, bucket_size=25, mode="generic", record_threshold=0.5):
        self.bucket_size = bucket_size
        self.mode = mode
        self.record_threshold = record_threshold
        self.counts = collections.Counter()
        self.n_recorded = 0
        self.n_filtered = 0

    def keys(self, canon_smiles):
        """Scaffold key per entry; None passes through unfiltered."""
        return [scaffold_key(s, self.mode) if s else None for s in canon_smiles]

    def multipliers(self, keys):
        out = []
        for k in keys:
            if k is None:
                out.append(1.0)
                continue
            m = 1.0 - self.counts[k] / self.bucket_size
            out.append(min(1.0, max(0.0, m)))
        return out

    def record(self, keys, p_active):
        """Count the molecules that actually cleared the bar this step.

        Deliberately counts duplicates: emitting the same molecule twenty
        times is exactly the behaviour the filter exists to make unprofitable,
        so each emission consumes bucket capacity.
        """
        for k, p in zip(keys, p_active):
            if k is None or p < self.record_threshold:
                continue
            self.counts[k] += 1
            self.n_recorded += 1

    def stats(self):
        n = len(self.counts)
        saturated = sum(1 for c in self.counts.values() if c >= self.bucket_size)
        top = self.counts.most_common(1)
        return {"n_buckets": n,
                "n_saturated": saturated,
                "top_bucket_count": top[0][1] if top else 0,
                "n_recorded": self.n_recorded}
