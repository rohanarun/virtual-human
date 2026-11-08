"""Render a graph (or a subgraph) to a diagram you can inspect.

Emits Mermaid (text, renders anywhere) and can rasterize to SVG/PNG with the
Mermaid CLI (`mmdc`) when present. "Render part of it to verify it partially" is
exactly `graph.subgraph(...)` followed by `to_mermaid(...)`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .core import Graph

# node styling by type (fill, stroke) in both light-ish and readable tones
_NODE_STYLE = {
    "blood": ("#C2373A", "#ffffff"),
    "compartment": ("#2B44C4", "#ffffff"),
    "reservoir": ("#C2373A", "#ffffff"),
    "junction": ("#586379", "#ffffff"),
    "sink": ("#586379", "#ffffff"),
    "lumen": ("#B26A12", "#ffffff"),
    "target": ("#1A8A54", "#ffffff"),
    "atom": ("#2B44C4", "#ffffff"),
    "species": ("#2B44C4", "#ffffff"),
    "reaction": ("#586379", "#ffffff"),
    "moiety": ("#1A8A54", "#ffffff"),
}
_EDGE_LABEL = {
    "flow": lambda e: f"{e.attrs.get('flow', 0):.1f} L/h",
    "conduit": lambda e: f"{e.attrs.get('flow_l_min', e.attrs.get('flow', 0)):.2f} L/min",
    "clears": lambda e: (f"CL {e.attrs.get('cl', 0):.2f} {e.attrs.get('units', '?')} "
                          f"({e.attrs.get('reference_basis', 'unspecified')})"),
    "absorbs": lambda e: f"Fa·Fg={e.attrs.get('fa',0)*e.attrs.get('fg',1):.2f}",
    "binds": lambda e: f"Kd {e.attrs.get('kd_nm','?')} nM",
    "bond": lambda e: "",
    # SBML reaction networks: show a coefficient only when it is not 1
    "reactant": lambda e: "" if str(e.attrs.get("stoichiometry", "1")) == "1" else str(e.attrs["stoichiometry"]),
    "product": lambda e: "" if str(e.attrs.get("stoichiometry", "1")) == "1" else str(e.attrs["stoichiometry"]),
    "modifier": lambda e: "modifies",
    "member": lambda e: "" if str(e.attrs.get("coefficient", "1")) == "1" else str(e.attrs["coefficient"]),
}


def to_mermaid(g: Graph, direction: str = "LR", edge_labels: bool = True, title: str | None = None) -> str:
    lines = [f"%% {title or g.name}", f"flowchart {direction}"]
    safe = lambda s: s.replace("-", "_").replace(" ", "_").replace(":", "_")  # noqa: E731
    def q(s):   # mermaid-safe quoted label
        return '"' + str(s).replace('"', "'") + '"'
    for n in g.nodes.values():
        lbl = q(n.label or n.id)
        shape = ("([{}])" if n.type in ("blood", "lumen") else
                 "[({})]" if n.type == "sink" else
                 "{{{}}}" if n.type == "target" else "[{}]")
        lines.append(f"  {safe(n.id)}{shape.format(lbl)}")
    for e in g.edges:
        lab = ""
        if edge_labels and e.type in _EDGE_LABEL:
            lab = _EDGE_LABEL[e.type](e)
        arrow = "-->" if e.directed else "---"
        conn = f'{arrow}|{q(lab)}|' if lab else arrow
        lines.append(f"  {safe(e.src)} {conn} {safe(e.dst)}")
    # class styling
    styles = {}
    for t, (fill, fg) in _NODE_STYLE.items():
        ids = [safe(n.id) for n in g.nodes.values() if n.type == t]
        if ids:
            cname = f"c_{t}"
            lines.append(f"  classDef {cname} fill:{fill},stroke:#222,color:{fg};")
            lines.append(f"  class {','.join(ids)} {cname};")
    return "\n".join(lines)


def render(g: Graph, out_path: str, direction: str = "LR", edge_labels: bool = True, title: str | None = None) -> str:
    """Write a diagram. If `out_path` ends in .svg/.png and `mmdc` is installed, it is
    rasterized; otherwise the Mermaid source is written to <out_path>.mmd."""
    mmd = to_mermaid(g, direction, edge_labels, title)
    out = Path(out_path)
    mmdc = shutil.which("mmdc")
    if out.suffix in (".svg", ".png") and mmdc:
        with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as f:
            f.write(mmd)
            src = f.name
        try:
            subprocess.run([mmdc, "-i", src, "-o", str(out), "-b", "transparent"],
                           check=True, capture_output=True, timeout=120)
            return str(out)
        except Exception:
            pass  # fall through to writing the source
    srcpath = out.with_suffix(".mmd")
    srcpath.write_text(mmd)
    return str(srcpath)
