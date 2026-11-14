"""DUPLO brick colour palette (configs/brick-palette.yaml). numpy + PyYAML only.

Perception names the colour of a real brick with ``nearest``. The sim twin and the training
data use ``albedo`` (the sim surface colour).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "brick-palette.yaml"


@dataclass(frozen=True)
class PaletteColor:
    name: str
    lego: str
    albedo: tuple[float, float, float]  # sim surface colour, 0..1
    observed: tuple[tuple[float, float, float], ...]  # real camera samples, 0..1
    seen: bool
    main: bool


@dataclass(frozen=True)
class Palette:
    colors: dict[str, PaletteColor]
    gain: float  # observed camera colour x gain = sim albedo

    def names(self, main_only: bool = False) -> list[str]:
        return [n for n, c in self.colors.items() if c.main or not main_only]

    def albedo(self, name: str) -> tuple[float, float, float]:
        return self.colors[name].albedo

    def references(self, name: str) -> np.ndarray:
        """Camera-colour references of one colour: the observed samples, else albedo / gain."""
        c = self.colors[name]
        refs = c.observed or (tuple(v / self.gain for v in c.albedo),)
        return np.asarray(refs, float)

    def nearest(self, rgb_observed) -> tuple[str, float]:
        """Name of the palette colour nearest to a camera colour (0..1), and the Lab distance."""
        lab = rgb_to_lab(np.asarray(rgb_observed, float).reshape(1, 3))[0]
        best, best_d = "", float("inf")
        for name in self.colors:
            d = float(np.min(np.linalg.norm(rgb_to_lab(self.references(name)) - lab, axis=1)))
            if d < best_d:
                best, best_d = name, d
        return best, best_d


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..1 (N, 3) -> CIE Lab (D65)."""
    rgb = np.clip(np.asarray(rgb, float), 0, 1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1
    )


def _rgb(value, where: str) -> tuple[float, float, float]:
    v = tuple(float(x) for x in value)
    if len(v) != 3 or not all(0 <= x <= 1 for x in v):
        raise ValueError(f"{where}: need [r, g, b] with numbers from 0 to 1, got {value!r}")
    return v


def parse_palette(raw: dict, source: str = "dict") -> Palette:
    gain = float(raw.get("gain", 1.08))
    colors = {}
    for name, entry in (raw.get("colors") or {}).items():
        where = f"{source}: colors.{name}"
        unknown = set(entry) - {"lego", "albedo", "observed", "seen", "main"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        colors[name] = PaletteColor(
            name=name,
            lego=str(entry.get("lego", name)),
            albedo=_rgb(entry["albedo"], f"{where}.albedo"),
            observed=tuple(_rgb(s, f"{where}.observed") for s in entry.get("observed") or []),
            seen=bool(entry.get("seen", False)),
            main=bool(entry.get("main", False)),
        )
    if not colors:
        raise ValueError(f"{source}: the palette has no colours")
    return Palette(colors=colors, gain=gain)


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> Palette:
    return parse_palette(yaml.safe_load(Path(path).read_text()), path)


def load_palette(path: str | Path | None = None) -> Palette:
    path = Path(path or DEFAULT_PATH)
    return _load(str(path), path.stat().st_mtime)
