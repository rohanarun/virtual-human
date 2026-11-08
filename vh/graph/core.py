"""The typed property graph: nodes, edges, schema, subgraphs, serialization.

Design choices:
  - Nodes and edges are *typed* and carry free-form attributes, so one graph can
    hold organs, molecules, reactions and atoms without a bespoke class per kind.
  - Every node carries a `scale` ("organism" / "organ" / "tissue" / "cell" /
    "molecule" / "atom"), so a multi-scale model is one graph you can slice by
    scale.
  - An edge may name a `model` -- a key into a registry of the real computational
    model behind it (a PBPK flow, a Rodgers-Rowland Kp, a quantum binding Kd).
    The graph stays declarative; the model is looked up when you execute a part.
  - A `Schema` states which node/edge types are legal and what attributes they
    must carry, plus structural invariants, so a formalization can be validated
    rather than trusted.

No third-party dependencies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable


@dataclass
class Node:
    id: str
    type: str
    scale: str = "unspecified"
    attrs: dict = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.id


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    directed: bool = True
    attrs: dict = field(default_factory=dict)
    model: str | None = None          # key into a model registry (the physics behind this edge)

    @property
    def key(self):
        return (self.src, self.dst, self.type)


class Graph:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[str, list[int]] = defaultdict(list)   # node -> edge indices (as src)
        self._in: dict[str, list[int]] = defaultdict(list)    # node -> edge indices (as dst)

    # --- construction --------------------------------------------------------
    def add_node(self, id: str, type: str, scale: str = "unspecified", label: str | None = None, **attrs) -> Node:
        if id in self.nodes:
            raise ValueError(f"duplicate node id {id!r}")
        n = Node(id=id, type=type, scale=scale, attrs=dict(attrs), label=label)
        self.nodes[id] = n
        return n

    def add_edge(self, src: str, dst: str, type: str, directed: bool = True, model: str | None = None, **attrs) -> Edge:
        for nid in (src, dst):
            if nid not in self.nodes:
                raise ValueError(f"edge references unknown node {nid!r}")
        e = Edge(src=src, dst=dst, type=type, directed=directed, attrs=dict(attrs), model=model)
        i = len(self.edges)
        self.edges.append(e)
        self._out[src].append(i)
        self._in[dst].append(i)
        if not directed:
            self._out[dst].append(i)
            self._in[src].append(i)
        return e

    # --- queries -------------------------------------------------------------
    def node(self, id: str) -> Node:
        return self.nodes[id]

    def nodes_of_type(self, *types: str) -> list[Node]:
        s = set(types)
        return [n for n in self.nodes.values() if n.type in s]

    def nodes_of_scale(self, *scales: str) -> list[Node]:
        s = set(scales)
        return [n for n in self.nodes.values() if n.scale in s]

    def edges_of_type(self, *types: str) -> list[Edge]:
        s = set(types)
        return [e for e in self.edges if e.type in s]

    def out_edges(self, nid: str, types: Iterable[str] | None = None) -> list[Edge]:
        ts = set(types) if types is not None else None
        res = []
        for i in self._out.get(nid, []):
            e = self.edges[i]
            if ts is not None and e.type not in ts:
                continue
            # for an undirected edge reached from dst, present it oriented away from nid
            res.append(e if e.src == nid else Edge(nid, e.src, e.type, e.directed, e.attrs, e.model))
        return res

    def successors(self, nid: str, types: Iterable[str] | None = None) -> list[str]:
        return [e.dst for e in self.out_edges(nid, types)]

    def in_edges(self, nid: str, types: Iterable[str] | None = None) -> list[Edge]:
        ts = set(types) if types is not None else None
        return [self.edges[i] for i in self._in.get(nid, []) if ts is None or self.edges[i].type in ts]

    def degree(self, nid: str) -> int:
        return len(self._out.get(nid, [])) + len(self._in.get(nid, []))

    # --- subgraphs -----------------------------------------------------------
    def subgraph(self, node_ids: Iterable[str], name: str | None = None) -> "Graph":
        """Induced subgraph on the given nodes (all edges with both ends inside).

        Nodes keep this graph's insertion order, not the order (or set-iteration
        order) of `node_ids`, so serialization and rendering are reproducible."""
        keep = set(node_ids)
        unknown = keep - self.nodes.keys()
        if unknown:
            raise KeyError(f"subgraph references unknown nodes {sorted(unknown)!r}")
        g = Graph(name or f"{self.name}:sub")
        for n in self.nodes.values():
            if n.id in keep:
                g.add_node(n.id, n.type, n.scale, n.label, **dict(n.attrs))
        for e in self.edges:
            if e.src in keep and e.dst in keep:
                g.add_edge(e.src, e.dst, e.type, e.directed, e.model, **dict(e.attrs))
        return g

    def edge_view(self, *edge_types: str, name: str | None = None) -> "Graph":
        """A graph with the same nodes but only edges of the given types (e.g. the
        blood-flow subgraph, or the bond subgraph). The basis for scale-specific
        algorithms."""
        g = Graph(name or f"{self.name}:{'+'.join(edge_types)}")
        for n in self.nodes.values():
            g.add_node(n.id, n.type, n.scale, n.label, **dict(n.attrs))
        s = set(edge_types)
        for e in self.edges:
            if e.type in s:
                g.add_edge(e.src, e.dst, e.type, e.directed, e.model, **dict(e.attrs))
        return g

    # --- validation ----------------------------------------------------------
    def validate(self, schema: "Schema") -> list[str]:
        return schema.check(self)

    # --- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name,
                "nodes": [asdict(n) for n in self.nodes.values()],
                "edges": [asdict(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls(d.get("name", "graph"))
        for n in d["nodes"]:
            g.add_node(n["id"], n["type"], n.get("scale", "unspecified"), n.get("label"), **n.get("attrs", {}))
        for e in d["edges"]:
            g.add_edge(e["src"], e["dst"], e["type"], e.get("directed", True), e.get("model"), **e.get("attrs", {}))
        return g

    def save_json(self, path):
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    def summary(self) -> str:
        from collections import Counter
        nt = Counter(n.type for n in self.nodes.values())
        et = Counter(e.type for e in self.edges)
        sc = Counter(n.scale for n in self.nodes.values())
        return (f"{self.name}: {len(self.nodes)} nodes {dict(nt)}, {len(self.edges)} edges {dict(et)}; "
                f"scales {dict(sc)}")


# ----------------------------------------------------------------------------
# schema: what a valid formalization looks like
# ----------------------------------------------------------------------------
@dataclass
class EdgeRule:
    type: str
    src_types: set            # allowed source node types
    dst_types: set            # allowed destination node types
    required_attrs: set = field(default_factory=set)


class Schema:
    """Declarative constraints + structural invariants for a graph."""

    def __init__(self, node_types: dict[str, set] | None = None, edge_rules: list[EdgeRule] | None = None):
        self.node_types = node_types or {}          # type -> required attribute names
        self.edge_rules = {r.type: r for r in (edge_rules or [])}
        self.invariants: list[tuple[str, Callable[[Graph], list[str]]]] = []

    def invariant(self, name: str):
        def deco(fn):
            self.invariants.append((name, fn))
            return fn
        return deco

    def add_invariant(self, name: str, fn: Callable[[Graph], list[str]]):
        self.invariants.append((name, fn))

    def check(self, g: Graph) -> list[str]:
        problems: list[str] = []
        for n in g.nodes.values():
            if self.node_types and n.type not in self.node_types:
                problems.append(f"node {n.id}: unknown type {n.type!r}")
                continue
            for a in self.node_types.get(n.type, set()):
                if a not in n.attrs:
                    problems.append(f"node {n.id} ({n.type}): missing attribute {a!r}")
        for e in g.edges:
            r = self.edge_rules.get(e.type)
            if self.edge_rules and r is None:
                problems.append(f"edge {e.key}: unknown type {e.type!r}")
                continue
            if r is not None:
                st, dt = g.nodes[e.src].type, g.nodes[e.dst].type
                if r.src_types and st not in r.src_types:
                    problems.append(f"edge {e.key}: source type {st!r} not allowed for {e.type!r}")
                if r.dst_types and dt not in r.dst_types:
                    problems.append(f"edge {e.key}: dest type {dt!r} not allowed for {e.type!r}")
                for a in r.required_attrs:
                    if a not in e.attrs:
                        problems.append(f"edge {e.key}: missing attribute {a!r}")
        for name, fn in self.invariants:
            for msg in fn(g):
                problems.append(f"[{name}] {msg}")
        return problems
