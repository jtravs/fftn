# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Catalogues: TOML files of weighted case groups (DESIGN.md 16.1; format in bench/README.md)."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CATALOGUE_DIR = Path(__file__).resolve().parent.parent / "catalogues"
TIERS = ("smoke", "quick", "tracking", "full")
KINDS = ("c2c", "r2c", "c2r")
GROUP_KEYS = {"name", "kind", "dims", "batch", "place", "direction", "weight", "tiers", "note"}


@dataclass(frozen=True)
class Case:
    kind: str
    dims: tuple[int, ...]
    batch: int
    place: str  # "in" or "out"
    sign: int  # -1 forward, +1 backward
    weight: float = 1.0
    tiers: frozenset[str] = field(default_factory=frozenset)
    group: str = ""

    @property
    def id(self) -> str:
        shape = "x".join(str(d) for d in self.dims)
        tail = "-bwd" if self.kind == "c2c" and self.sign > 0 else ""
        return f"{self.kind}-{shape}-b{self.batch}-{'ip' if self.place == 'in' else 'op'}{tail}"

    @property
    def points(self) -> int:
        return math.prod(self.dims)

    @property
    def flops(self) -> float:
        """The benchFFT convention: 5 n log2 n per complex transform, half that for real."""
        n = self.points
        per = (5.0 if self.kind == "c2c" else 2.5) * n * math.log2(n) if n > 1 else 0.0
        return per * self.batch

    def line(self) -> str:
        """The harness's case line."""
        dims = "x".join(str(d) for d in self.dims)
        return (
            f"id={self.id} kind={self.kind} dims={dims} batch={self.batch} "
            f"place={self.place} sign={self.sign}"
        )


@dataclass(frozen=True)
class Catalogue:
    name: str
    path: Path
    sha256: str
    cases: tuple[Case, ...]

    def select(self, tier: str | None = None, pattern: str | None = None) -> list[Case]:
        """Cases of a tier (every case is in "full"), whose id matches the regex."""
        if tier is not None and tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; tiers are {', '.join(TIERS)}")
        rx = re.compile(pattern) if pattern else None
        return [
            c
            for c in self.cases
            if (tier in (None, "full") or tier in c.tiers) and (rx is None or rx.search(c.id))
        ]


def resolve(name_or_path: str) -> Path:
    """A catalogue path, or the name of one under bench/catalogues/."""
    p = Path(name_or_path)
    if p.suffix == ".toml" or p.exists():
        return p
    return CATALOGUE_DIR / f"{name_or_path}.toml"


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, list) else [value]


def _dims(value: object, where: str) -> tuple[int, ...]:
    if isinstance(value, int):
        dims: tuple[int, ...] = (value,)
    elif isinstance(value, str) and re.fullmatch(r"\d+(x\d+){0,2}", value):
        dims = tuple(int(x) for x in value.split("x"))
    else:
        raise ValueError(f"{where}: dims entries are lengths or strings like '64x48': {value!r}")
    if any(d < 1 for d in dims):
        raise ValueError(f"{where}: lengths must be positive: {value!r}")
    return dims


def load(name_or_path: str) -> Catalogue:
    path = resolve(name_or_path)
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    head = data.get("catalogue", {})
    cases: dict[str, Case] = {}
    for i, group in enumerate(data.get("group", [])):
        where = f"{path.name} group {group.get('name', i)}"
        unknown = set(group) - GROUP_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        tiers = frozenset(_as_list(group.get("tiers", [])))
        if not tiers <= set(TIERS):
            raise ValueError(f"{where}: tiers must be among {TIERS}")
        weight = float(group.get("weight", 1.0))
        product = itertools.product(
            _as_list(group.get("kind", "c2c")),
            [_dims(d, where) for d in _as_list(group["dims"])],
            _as_list(group.get("batch", 1)),
            _as_list(group.get("place", "out")),
            _as_list(group.get("direction", "fwd")),
        )
        for kind, dims, batch, place, direction in product:
            if kind not in KINDS or place not in ("in", "out") or direction not in ("fwd", "bwd"):
                raise ValueError(f"{where}: bad kind, place or direction")
            if kind != "c2c" and direction != "fwd":
                raise ValueError(f"{where}: direction applies to c2c only")
            sign = {"r2c": -1, "c2r": 1}.get(kind, -1 if direction == "fwd" else 1)
            case = Case(kind, dims, int(batch), place, sign, weight, tiers, str(group.get("name")))
            old = cases.get(case.id)
            if old is not None:
                if old.weight != case.weight:
                    raise ValueError(f"{where}: {case.id} appears with two weights")
                case = Case(
                    kind, dims, int(batch), place, sign, weight, old.tiers | tiers, old.group
                )
            cases[case.id] = case
    return Catalogue(
        str(head.get("name", path.stem)),
        path,
        hashlib.sha256(raw).hexdigest(),
        tuple(cases.values()),
    )
