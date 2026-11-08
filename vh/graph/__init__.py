"""A typed property graph for formalizing the body across scales.

The body is a graph at every scale: organs joined by blood flow, molecules joined
by reactions, atoms joined by bonds. This package gives one representation for all
of them -- typed nodes and edges that carry attributes and, where it matters, a
reference to the validated computational model behind an edge (a blood flow, a
partition coefficient, a binding constant). So the graph is both a *map* of the
system and a scaffold that is partially *executable* and *verifiable*.

What this buys (see docs/graph_formalization.md):
  - structural questions answered exactly by graph algorithms (reachability,
    pathways, cycles, flow balance, bottlenecks);
  - partial verification: pull out a subgraph and check it on its own;
  - a place to wire the physics engines (vh.quantum, vh.molecular, vh.body)
    together and route information between scales.

ModelRegistry executes an explicitly registered Edge.model with scalar interface
contracts. See docs/model_contracts.md; existing engine adapters are not registered
automatically, and graph reachability is not a claim of biological causation.

For explicit elementary reaction networks and ordered interventions, import
vh.graph.reactions (NumPy required); see docs/reaction_experiments.md. This module
is intentionally not imported here, keeping the basic graph API dependency-free.
"""

from .core import Edge, Graph, Node, Schema  # noqa: F401
from .contracts import ContractError, ModelRegistry, ModelSpec, PortSpec, Quantity  # noqa: F401
from . import algorithms  # noqa: F401
