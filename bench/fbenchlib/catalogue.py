# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
"""Catalogues: TOML files of weighted case groups (DESIGN.md 16.1; format in bench/README.md)."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

CATALOGUE_DIR = Path(__file__).resolve().parent.parent / "catalogues"
TIERS = ("smoke", "quick", "tracking", "full")
KINDS = ("c2c", "r2c", "c2r")
PRECISIONS = ("f64", "f32")
GROUP_KEYS = {
    "name",
    "kind",
    "dims",
    "batch",
    "place",
    "direction",
    "precision",
    "layout",
    "outer",
    "weight",
    "tiers",
    "note",
}


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
    precision: str = "f64"
    layout: str = "tight"  # or "interleaved": transforms one element apart (bench/README.md)
    outer: int = 1  # blocks of interleaved transforms

    @property
    def id(self) -> str:
        shape = "x".join(str(d) for d in self.dims)
        tail = "-bwd" if self.kind == "c2c" and self.sign > 0 else ""
        tail += "-f32" if self.precision == "f32" else ""
        if self.layout == "interleaved":
            tail += f"-il-o{self.outer}"
        return f"{self.kind}-{shape}-b{self.batch}-{'ip' if self.place == 'in' else 'op'}{tail}"

    @property
    def points(self) -> int:
        return math.prod(self.dims)

    @property
    def flops(self) -> float:
        """The benchFFT convention: 5 n log2 n per complex transform, half that for real."""
        n = self.points
        per = (5.0 if self.kind == "c2c" else 2.5) * n * math.log2(n) if n > 1 else 0.0
        return per * self.batch * self.outer

    def line(self) -> str:
        """The harness's case line. The precision is written only when it is f32, so that
        double-precision cases still run on harnesses older than the key."""
        dims = "x".join(str(d) for d in self.dims)
        prec = " prec=f32" if self.precision == "f32" else ""
        layout = f" layout=interleaved outer={self.outer}" if self.layout == "interleaved" else ""
        return (
            f"id={self.id} kind={self.kind} dims={dims} batch={self.batch} "
            f"place={self.place} sign={self.sign}{prec}{layout}"
        )


@dataclass(frozen=True)
class Catalogue:
    name: str
    path: Path
    sha256: str
    cases: tuple[Case, ...]

    def select(
        self, tier: str | None = None, pattern: str | None = None, precision: str | None = None
    ) -> list[Case]:
        """Cases of a tier (every case is in "full"), whose id matches the regex. With a
        precision, every case of the tier is taken in that precision first (a case the
        catalogue already lists in it is kept once), and the regex sees the new ids."""
        if tier is not None and tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; tiers are {', '.join(TIERS)}")
        if precision is not None and precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        rx = re.compile(pattern) if pattern else None
        chosen: dict[str, Case] = {}
        for c in self.cases:
            if tier not in (None, "full") and tier not in c.tiers:
                continue
            if precision is not None:
                c = replace(c, precision=precision)
            if (rx is None or rx.search(c.id)) and c.id not in chosen:
                chosen[c.id] = c
        return list(chosen.values())


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
            _as_list(group.get("precision", "f64")),
            _as_list(group.get("outer", 1)),
        )
        layout = str(group.get("layout", "tight"))
        if layout not in ("tight", "interleaved"):
            raise ValueError(f"{where}: layout is tight or interleaved")
        for kind, dims, batch, place, direction, precision, outer in product:
            if kind not in KINDS or place not in ("in", "out") or direction not in ("fwd", "bwd"):
                raise ValueError(f"{where}: bad kind, place or direction")
            if precision not in PRECISIONS:
                raise ValueError(f"{where}: precision is one of {PRECISIONS}")
            if kind != "c2c" and direction != "fwd":
                raise ValueError(f"{where}: direction applies to c2c only")
            sign = {"r2c": -1, "c2r": 1}.get(kind, -1 if direction == "fwd" else 1)
            name = str(group.get("name"))
            case = Case(
                kind,
                dims,
                int(batch),
                place,
                sign,
                weight,
                tiers,
                name,
                precision,
                layout,
                int(outer),
            )
            old = cases.get(case.id)
            if old is not None:
                if old.weight != case.weight:
                    raise ValueError(f"{where}: {case.id} appears with two weights")
                case = Case(
                    kind,
                    dims,
                    int(batch),
                    place,
                    sign,
                    weight,
                    old.tiers | tiers,
                    old.group,
                    precision,
                    layout,
                    int(outer),
                )
            cases[case.id] = case
    return Catalogue(
        str(head.get("name", path.stem)),
        path,
        hashlib.sha256(raw).hexdigest(),
        tuple(cases.values()),
    )
