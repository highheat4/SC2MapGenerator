"""Milestone 6: competitive-plausibility filter.

With only ~350 (here ~109) complete maps, V1 uses simple statistics rather than a learned
model (spec section 10). The corpus is a source of empirical distributions and an outlier
detector: a generated map should look like it was *drawn from the same design space*, but the
runtime acceptance band is deliberately WIDER than the training-set center so unusual-but-valid
maps survive (that is the whole point of "controlled weirdness").

Two complementary signals, both computed on the standard feature vector (``features.py``):

  * Per-feature bands. For each tracked scalar we take the corpus [p_lo, p_hi] percentile
    range and broaden it by a tolerance factor (and by the weirdness knob). A map is
    implausible if more than ``max_outliers`` features fall outside their broadened band; the
    offending features are reported so the reason is legible.
  * Nearest-neighbour distance. In robustly-standardized feature space we measure the
    Euclidean distance to the closest corpus map. A large NN distance means "unlike anything
    in the corpus" even when no single feature is extreme.

Use :func:`PlausibilityModel.from_features` to fit on the corpus once, then
:meth:`PlausibilityModel.score` per generated map (feature dict from ``extract_features``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Tracked features: geometry / topology quantities that genuinely vary between maps and are
# reliably computable for a generated MapIR. Resource counts (constant by construction) and
# symmetry (already a hard/soft check in the validator) are deliberately excluded.
DEFAULT_FEATURES: tuple[str, ...] = (
    "playable_ratio",
    "aspect",
    "n_bases",
    "base_density",
    "n_elevation_levels",
    "high_ground_ratio",
    "main_to_natural_mean",
    "rush_distance_min",
    "base_to_base_mean",
    "n_ramps",
)


@dataclass
class PlausibilityConfig:
    features: tuple[str, ...] = DEFAULT_FEATURES
    pct: tuple[float, float] = (5.0, 95.0)   # corpus percentile range each band is built on
    tol: float = 1.6                         # broaden the band's half-span by this factor
    max_outliers: int = 2                    # allow this many features outside their band
    nn_max: float = 6.0                      # NN distance (std units) beyond which -> implausible
    weirdness: float = 0.0                   # extra broadening (matches the generator knob)


@dataclass
class Band:
    lo: float
    hi: float
    center: float
    scale: float                             # robust half-span, for z / NN standardization


@dataclass
class PlausibilityReport:
    plausible: bool
    outliers: list[str] = field(default_factory=list)
    distance: float = 0.0                     # mean robust deviation across tracked features
    nn_distance: float = 0.0                  # standardized NN distance to closest corpus map
    per_feature: dict = field(default_factory=dict)

    def summary(self) -> str:
        tag = "PLAUSIBLE" if self.plausible else "IMPLAUSIBLE"
        return (f"{tag}: {len(self.outliers)} outlier(s), "
                f"dist={self.distance:.2f}, nn={self.nn_distance:.2f}")


class PlausibilityModel:
    def __init__(self, bands: dict[str, Band], corpus_std: np.ndarray,
                 cfg: PlausibilityConfig):
        self.bands = bands
        self.corpus_std = corpus_std          # (n_maps, n_features) standardized
        self.cfg = cfg
        self.features = tuple(f for f in cfg.features if f in bands)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_features(cls, feats: list[dict],
                      cfg: PlausibilityConfig | None = None) -> "PlausibilityModel":
        cfg = cfg or PlausibilityConfig()
        p_lo, p_hi = cfg.pct
        broaden = cfg.tol * (1.0 + cfg.weirdness)
        bands: dict[str, Band] = {}
        cols: list[np.ndarray] = []
        used: list[str] = []
        for name in cfg.features:
            vals = np.array(
                [f[name] for f in feats if isinstance(f.get(name), (int, float))],
                dtype=float,
            )
            if vals.size < 4:
                continue
            lo, hi = np.percentile(vals, [p_lo, p_hi])
            center = float(np.median(vals))
            span = float(hi - lo)
            scale = max(span / 2.0, 1e-6)
            margin = (span / 2.0) * (broaden - 1.0)
            bands[name] = Band(lo=float(lo) - margin, hi=float(hi) + margin,
                               center=center, scale=scale)
            used.append(name)
            cols.append(vals)

        # standardized corpus matrix for NN (only maps with all used features present)
        rows = []
        for f in feats:
            if all(isinstance(f.get(n), (int, float)) for n in used):
                rows.append([(f[n] - bands[n].center) / bands[n].scale for n in used])
        corpus_std = np.array(rows, dtype=float) if rows else np.zeros((0, len(used)))
        model = cls(bands, corpus_std, cfg)
        model.features = tuple(used)
        return model

    @classmethod
    def from_file(cls, path: str | Path,
                  cfg: PlausibilityConfig | None = None) -> "PlausibilityModel":
        data = json.loads(Path(path).read_text())
        feats = data if isinstance(data, list) else data.get("features", [])
        return cls.from_features(feats, cfg)

    # ------------------------------------------------------------------ #
    def score(self, feats: dict) -> PlausibilityReport:
        outliers: list[str] = []
        devs: list[float] = []
        per: dict = {}
        vec: list[float] = []
        ok_vec = True
        for name in self.features:
            band = self.bands[name]
            v = feats.get(name)
            if not isinstance(v, (int, float)):
                per[name] = None
                ok_vec = False
                continue
            v = float(v)
            z = abs(v - band.center) / band.scale
            devs.append(z)
            inside = band.lo <= v <= band.hi
            per[name] = {"value": round(v, 3), "band": (round(band.lo, 2), round(band.hi, 2)),
                         "z": round(z, 2), "inside": inside}
            if not inside:
                outliers.append(f"{name}={v:.2f} outside [{band.lo:.2f},{band.hi:.2f}]")
            vec.append((v - band.center) / band.scale)

        distance = float(np.mean(devs)) if devs else 0.0
        nn = float("inf")
        if ok_vec and self.corpus_std.shape[0] and len(vec) == self.corpus_std.shape[1]:
            d = np.linalg.norm(self.corpus_std - np.array(vec), axis=1)
            nn = float(d.min())
        elif not ok_vec:
            nn = float("inf")

        plausible = len(outliers) <= self.cfg.max_outliers and nn <= self.cfg.nn_max
        return PlausibilityReport(
            plausible=plausible,
            outliers=outliers,
            distance=round(distance, 3),
            nn_distance=round(nn, 3) if math.isfinite(nn) else nn,
            per_feature=per,
        )
