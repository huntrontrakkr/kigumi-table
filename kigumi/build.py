"""One call from parameters to finished solids, relations and a report."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .joinery import Relation, apply, relations
from .layout import Layout, build_layout
from .model import Part, make_parts
from .params import Params


@dataclass
class Design:
    p: Params
    layout: Layout
    parts: list[Part]
    rels: list[Relation]
    timings: dict = field(default_factory=dict)

    def part(self, pid: str) -> Part:
        for q in self.parts:
            if q.id == pid:
                return q
        raise KeyError(pid)


def build(p: Params) -> Design:
    t = {}
    p = p.effective()
    t0 = time.perf_counter()
    lay = build_layout(p)
    p = lay.p                  # the layout may have turned the grid (orient = "load")
    t["layout"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    parts = make_parts(lay)
    rels = relations(parts, p)
    t["relations"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    apply(parts, rels, p)
    t["solids"] = time.perf_counter() - t0
    return Design(p, lay, parts, rels, t)
