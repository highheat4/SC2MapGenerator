"""Distributions the skeleton generator samples from.

Design intent (spec sections 6, 8, 10):
    * Hard constraints (symmetry, 2 starts, min spacing) live in the generator/validator.
    * *Parameters* (dimensions, base count, natural offset, spacing) are sampled here.
    * The competitive corpus is the CENTER of the design space, not its boundary, so a
      ``weirdness`` factor broadens the learned ranges without inventing tile noise.

``DefaultPriors`` ships hardcoded competitive-ish ranges so the generator runs today.
``LearnedPriors.from_features`` builds empirical distributions from the Milestone 2
feature dicts (see ``sc2mapgen.features.extract_features``) and delegates any quantity it
can't learn (e.g. main-corner placement, natural angle) back to ``DefaultPriors``.

All sampling methods take an explicit ``numpy`` Generator for reproducibility (per-match
generation must be seedable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sc2mapgen.ir import Rect


@dataclass
class DimSample:
    grid_w: int
    grid_h: int
    playable: Rect


class Priors(Protocol):
    def sample_dims(self, rng: np.random.Generator) -> DimSample: ...
    def sample_total_bases(self, rng: np.random.Generator) -> int: ...
    def sample_main_fraction(self, rng: np.random.Generator) -> tuple[float, float]: ...
    def sample_natural_offset(self, rng: np.random.Generator) -> tuple[float, float]: ...
    def sample_min_spacing(self, rng: np.random.Generator) -> float: ...


def _widen(lo: float, hi: float, weirdness: float) -> tuple[float, float]:
    """Broaden a [lo, hi] band symmetrically about its midpoint by ``weirdness``."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * (1.0 + weirdness)
    return mid - half, mid + half


@dataclass
class DefaultPriors:
    """Sensible competitive ranges (tiles), usable before any corpus is analyzed."""

    weirdness: float = 0.0

    # playable area (tiles); grid = playable + 2*border
    playable_min: int = 116
    playable_max: int = 156
    border: int = 18
    aspect_jitter: float = 0.08  # height relative to width

    # total bases on the map (includes 2 mains + 2 naturals); rest are generic
    base_count_choices: tuple[int, ...] = (8, 10, 12, 14, 16)
    base_count_weights: tuple[float, ...] = (0.12, 0.28, 0.30, 0.22, 0.08)

    # main placement as a fraction into the playable rect from its near corner
    main_frac_lo: float = 0.10
    main_frac_hi: float = 0.19

    # natural relative to main: distance (tiles) and angular offset from the
    # main->center direction (degrees)
    natural_dist_lo: float = 18.0
    natural_dist_hi: float = 30.0
    natural_angle_deg: float = 22.0

    # minimum center-to-center spacing between any two bases (tiles)
    spacing_lo: float = 16.0
    spacing_hi: float = 22.0

    def sample_dims(self, rng: np.random.Generator) -> DimSample:
        lo, hi = _widen(self.playable_min, self.playable_max, self.weirdness)
        pw = int(round(rng.uniform(lo, hi)))
        ph = int(round(pw * (1.0 + rng.uniform(-self.aspect_jitter, self.aspect_jitter))))
        # keep everything even so 180-degree rotation about the center lands on tiles
        pw += pw % 2
        ph += ph % 2
        grid_w = pw + 2 * self.border
        grid_h = ph + 2 * self.border
        playable = Rect(x=self.border, y=self.border, width=pw, height=ph)
        return DimSample(grid_w=grid_w, grid_h=grid_h, playable=playable)

    def sample_total_bases(self, rng: np.random.Generator) -> int:
        w = np.array(self.base_count_weights, dtype=float)
        w = w / w.sum()
        return int(rng.choice(self.base_count_choices, p=w))

    def sample_main_fraction(self, rng: np.random.Generator) -> tuple[float, float]:
        lo, hi = _widen(self.main_frac_lo, self.main_frac_hi, self.weirdness)
        return float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))

    def sample_natural_offset(self, rng: np.random.Generator) -> tuple[float, float]:
        lo, hi = _widen(self.natural_dist_lo, self.natural_dist_hi, self.weirdness)
        dist = float(rng.uniform(lo, hi))
        amax = self.natural_angle_deg * (1.0 + self.weirdness)
        angle = math.radians(float(rng.uniform(-amax, amax)))
        return dist, angle

    def sample_min_spacing(self, rng: np.random.Generator) -> float:
        lo, hi = _widen(self.spacing_lo, self.spacing_hi, self.weirdness)
        return float(rng.uniform(lo, hi))


@dataclass
class LearnedPriors:
    """Empirical priors bootstrapped from Milestone 2 feature dicts.

    Learns the quantities the corpus actually measures (dimensions, base count, natural
    distance, base spacing) and delegates the rest to an internal ``DefaultPriors``.
    """

    dims: list[tuple[int, int]] = field(default_factory=list)          # (playable_w, playable_h)
    total_bases: list[int] = field(default_factory=list)
    natural_dist: list[float] = field(default_factory=list)
    base_to_base_min: list[float] = field(default_factory=list)
    weirdness: float = 0.0
    _fallback: DefaultPriors = field(default_factory=DefaultPriors)

    @classmethod
    def from_features(cls, features: list[dict], weirdness: float = 0.0) -> "LearnedPriors":
        dims, totals, nat, bbmin = [], [], [], []
        for f in features:
            if f.get("playable_width") and f.get("playable_height"):
                dims.append((int(f["playable_width"]), int(f["playable_height"])))
            if f.get("n_bases"):
                totals.append(int(f["n_bases"]))
            if f.get("main_to_natural_mean"):
                nat.append(float(f["main_to_natural_mean"]))
            if f.get("base_to_base_min"):
                bbmin.append(float(f["base_to_base_min"]))
        fb = DefaultPriors(weirdness=weirdness)
        return cls(
            dims=dims,
            total_bases=totals,
            natural_dist=nat,
            base_to_base_min=bbmin,
            weirdness=weirdness,
            _fallback=fb,
        )

    def _jitter(self, rng: np.random.Generator, v: float, frac: float = 0.08) -> float:
        spread = frac * (1.0 + self.weirdness)
        return v * (1.0 + rng.uniform(-spread, spread))

    def sample_dims(self, rng: np.random.Generator) -> DimSample:
        if not self.dims:
            return self._fallback.sample_dims(rng)
        pw0, ph0 = self.dims[rng.integers(len(self.dims))]  # joint sample keeps aspect
        pw = int(round(self._jitter(rng, pw0)))
        ph = int(round(self._jitter(rng, ph0)))
        pw += pw % 2
        ph += ph % 2
        b = self._fallback.border
        return DimSample(grid_w=pw + 2 * b, grid_h=ph + 2 * b, playable=Rect(b, b, pw, ph))

    def sample_total_bases(self, rng: np.random.Generator) -> int:
        if not self.total_bases:
            return self._fallback.sample_total_bases(rng)
        n = int(self.total_bases[rng.integers(len(self.total_bases))])
        return max(6, n - (n % 2))  # keep even for clean 2-fold symmetry

    def sample_main_fraction(self, rng: np.random.Generator) -> tuple[float, float]:
        return self._fallback.sample_main_fraction(rng)

    def sample_natural_offset(self, rng: np.random.Generator) -> tuple[float, float]:
        if not self.natural_dist:
            return self._fallback.sample_natural_offset(rng)
        d = self._jitter(rng, float(self.natural_dist[rng.integers(len(self.natural_dist))]))
        _, angle = self._fallback.sample_natural_offset(rng)
        return d, angle

    def sample_min_spacing(self, rng: np.random.Generator) -> float:
        if not self.base_to_base_min:
            return self._fallback.sample_min_spacing(rng)
        return float(np.median(self.base_to_base_min)) * (1.0 - 0.1 * self.weirdness)
