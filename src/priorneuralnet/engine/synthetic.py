"""A realistic p >> n synthetic benchmark with a known ground-truth prior.

N samples (default 240) x P features (default 300): 15 named biological
pathways of 18 co-sampled features each, plus 30 unmapped orphan features.
Exactly 3 pathways drive the target linearly; all other features are
background noise. The target is a Bernoulli draw through a sigmoid of the
pathway mixture — hard, but learnable — and the ground-truth active pathway
names are returned so tests can verify concept recovery from above.

The signal is linear in the features by construction, which keeps the
attribution claim checkable: Integrated Gradients over a linear path is
exact, so attributing the injected pathways to the top of the concept
ranking is a hard correctness property, not a soft expectation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PATHWAY_NAMES: tuple[str, ...] = (
    "cell_cycle_checkpoint",
    "dna_damage_repair",
    "immune_signaling",
    "angiogenesis",
    "glycolysis_metabolism",
    "lipid_signaling",
    "circadian_entrainment",
    "cytoskeleton_remodeling",
    "chromatin_remodeling",
    "unfolded_protein_response",
    "hypoxia_response",
    "oxidative_stress_response",
    "apoptosis_regulation",
    "growth_factor_kinase",
    "interferon_response",
)


@dataclass
class SyntheticDataset:
    df: pd.DataFrame
    mapping: dict[str, str]  # feature -> pathway (orphans excluded by design)
    active_pathways: list[str]
    target_column: str = "target"
    seed: int = 0

    def save(self, out_dir: str | Path) -> dict[str, str]:
        """Write dataset.csv + prior.json + ground_truth.json; returns written paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data_path = out / "dataset.csv"
        prior_path = out / "prior.json"
        truth_path = out / "ground_truth.json"
        self.df.to_csv(data_path, index=False)
        prior_path.write_text(json.dumps(self.mapping, indent=1), encoding="utf-8")
        truth_path.write_text(
            json.dumps(
                {
                    "active_pathways": sorted(self.active_pathways),
                    "target_column": self.target_column,
                    "seed": self.seed,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return {
            "dataset": str(data_path),
            "prior": str(prior_path),
            "ground_truth": str(truth_path),
        }


def make_synthetic_prior_dataset(
    n_samples: int = 240,
    n_pathways: int = 15,
    pathway_size: int = 18,
    n_orphans: int = 30,
    n_active: int = 3,
    signal_strength: float = 1.5,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a p >> n tabular dataset with a named-pathway ground truth."""
    if n_active > n_pathways:
        raise ValueError("n_active must be <= n_pathways")
    rng = np.random.default_rng(seed)
    pathways = list(PATHWAY_NAMES[:n_pathways])
    active = sorted(rng.choice(pathways, size=n_active, replace=False).tolist())

    columns: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    z = np.zeros(n_samples)
    for pathway in pathways:
        # Co-sampled block: one latent pathway factor measured redundantly by
        # every member feature + measurement noise. This redundancy is exactly
        # what lets a concept layer denoise and dominate a flat MLP in p >> n.
        factor = rng.normal(0.0, 1.0, size=n_samples)
        loadings = rng.uniform(0.5, 1.0, size=pathway_size)
        block = factor[:, None] * loadings[None, :] + rng.normal(
            0.0, 0.5, size=(n_samples, pathway_size)
        )
        weights = rng.uniform(0.3, 1.0, size=pathway_size)
        for j in range(pathway_size):
            name = f"{pathway}_S{j + 1:03d}"
            columns[name] = block[:, j]
            mapping[name] = pathway
        if pathway in active:
            z += block @ weights
    for j in range(n_orphans):
        columns[f"unmapped_S{j + 1:03d}"] = rng.normal(0.0, 1.0, size=n_samples)

    z /= z.std()
    logits = signal_strength * z
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    df = pd.DataFrame(columns)
    df["target"] = y
    return SyntheticDataset(df=df, mapping=mapping, active_pathways=active, seed=seed)
