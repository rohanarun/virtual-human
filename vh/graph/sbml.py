"""Read a declared subset of SBML into an executable reaction network and a typed graph.

Written from scratch on the standard-library XML parser; libSBML/libRoadRunner
are used only as independent referees in the tests. Supported: SBML Level 2
(versions 1-5) and Level 3 (versions 1-2) core models built from

  compartments (constant size), species, global and local parameters,
  function definitions, initial assignments, assignment rules, and reactions
  with reactants, products, modifiers and a MathML kinetic law.

Anything outside that subset -- events, rate and algebraic rules, constraints,
delays, fast reactions, stoichiometryMath, conversion factors, variable
compartments, required packages, or a MathML element not listed in
``_OPERATORS`` -- raises ``UnsupportedSBML`` naming the construct. Nothing is
silently dropped except notes, annotations, units and SBO terms. Units are not
checked: the numbers are used as written, as an SBML simulator would.

SBML semantics implemented here: a species symbol in math means its
concentration (amount / compartment size) unless ``hasOnlySubstanceUnits``; a
kinetic law is in substance per time, so ``d amount_i / dt = sum_j N_ij v_j``;
boundary-condition species are not changed by reactions.

Documents containing a DOCTYPE are refused (no entity expansion); MathML is
compiled to closures over a fixed operator table, never to ``eval``.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import stoichiometry as S
from .core import EdgeRule, Graph, Schema

MATHML = "http://www.w3.org/1998/Math/MathML"
_TIME = "http://www.sbml.org/sbml/symbols/time"
_AVOGADRO = "http://www.sbml.org/sbml/symbols/avogadro"
_DELAY = "http://www.sbml.org/sbml/symbols/delay"


class SBMLError(ValueError):
    """The document is malformed or violates SBML semantics this reader relies on."""


class UnsupportedSBML(SBMLError):
    """The document is valid SBML but uses a construct outside the supported subset."""


# ---------------------------------------------------------------------------
# MathML -> expression tree
# ---------------------------------------------------------------------------
# A node is a tuple: ("num", float) | ("sym", id) | ("time",) | ("op", name, args)
# | ("piecewise", ((value, cond), ...), otherwise_or_None) | ("call", fname, args)

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ieee_div(a, b):
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _ieee_pow(a, b):
    try:
        r = a ** b
    except ZeroDivisionError:
        return math.inf
    except OverflowError:
        return math.inf
    return math.nan if isinstance(r, complex) else r


def _ieee(fn):
    """Domain errors become NaN and overflow becomes inf, as in IEEE arithmetic,
    so a kinetic law evaluated off its domain does not abort an integration."""
    def wrapped(args):
        try:
            return fn(args)
        except (ValueError, ZeroDivisionError):
            return math.nan
        except OverflowError:
            return math.inf
    return wrapped


def _log(args):
    return math.log10(args[0]) if len(args) == 1 else math.log(args[1], args[0])


def _chain(cmp):
    return lambda args: all(cmp(a, b) for a, b in zip(args, args[1:]))


_OPERATORS = {
    "plus": lambda a: sum(a) if a else 0.0,
    "minus": lambda a: -a[0] if len(a) == 1 else a[0] - a[1],
    "times": lambda a: math.prod(a) if a else 1.0,
    "divide": lambda a: _ieee_div(a[0], a[1]),
    "power": lambda a: _ieee_pow(a[0], a[1]),
    "root": _ieee(lambda a: _ieee_pow(a[1], 1.0 / a[0]) if len(a) == 2 else math.sqrt(a[0])),
    "exp": _ieee(lambda a: math.exp(a[0])),
    "ln": _ieee(lambda a: math.log(a[0])),
    "log": _ieee(_log),
    "abs": lambda a: abs(a[0]),
    "floor": lambda a: float(math.floor(a[0])),
    "ceiling": lambda a: float(math.ceil(a[0])),
    "sin": lambda a: math.sin(a[0]), "cos": lambda a: math.cos(a[0]),
    "tan": lambda a: math.tan(a[0]), "arcsin": _ieee(lambda a: math.asin(a[0])),
    "arccos": _ieee(lambda a: math.acos(a[0])), "arctan": lambda a: math.atan(a[0]),
    "sinh": _ieee(lambda a: math.sinh(a[0])), "cosh": _ieee(lambda a: math.cosh(a[0])),
    "tanh": lambda a: math.tanh(a[0]),
    "eq": _chain(lambda x, y: x == y), "neq": lambda a: a[0] != a[1],
    "gt": _chain(lambda x, y: x > y), "lt": _chain(lambda x, y: x < y),
    "geq": _chain(lambda x, y: x >= y), "leq": _chain(lambda x, y: x <= y),
    "and": lambda a: all(a), "or": lambda a: any(a), "xor": lambda a: sum(map(bool, a)) % 2 == 1,
    "not": lambda a: not a[0],
}
_ARITY = {"minus": (1, 2), "divide": (2, 2), "power": (2, 2), "neq": (2, 2), "not": (1, 1)}
_UNARY = {"exp", "ln", "abs", "floor", "ceiling", "sin", "cos", "tan", "arcsin", "arccos",
          "arctan", "sinh", "cosh", "tanh"}
_CONSTANTS = {"pi": math.pi, "exponentiale": math.e, "true": 1.0, "false": 0.0,
              "infinity": math.inf, "notanumber": math.nan}
_INFIX = {"plus": "+", "minus": "-", "times": "*", "divide": "/", "power": "^"}


def _cn(el) -> tuple:
    kind = el.attrib.get("type", "real")
    parts = [p.strip() for p in "".join(el.itertext()).split()] if kind not in ("e-notation", "rational") else None
    if kind in ("e-notation", "rational"):
        seps = [c for c in el if _local(c.tag) == "sep"]
        if len(seps) != 1:
            raise SBMLError(f"<cn type={kind!r}> needs exactly one <sep/>")
        a = (el.text or "").strip()
        b = (seps[0].tail or "").strip()
        return ("num", float(a) * 10.0 ** float(b) if kind == "e-notation" else float(Fraction(a) / Fraction(b)))
    if kind not in ("real", "integer", "double"):
        raise UnsupportedSBML(f"<cn type={kind!r}>")
    if len(parts) != 1:
        raise SBMLError("<cn> must contain one number")
    return ("num", float(parts[0]))


def parse_mathml(el) -> tuple:
    """Expression tree of one MathML element (``<math>`` or a child)."""
    tag = _local(el.tag)
    if tag in ("math", "semantics"):
        kids = [c for c in el if _local(c.tag) not in ("annotation", "annotation-xml")]
        if len(kids) != 1:
            raise SBMLError(f"<{tag}> must contain exactly one expression")
        return parse_mathml(kids[0])
    if tag == "cn":
        return _cn(el)
    if tag == "ci":
        return ("sym", "".join(el.itertext()).strip())
    if tag == "csymbol":
        url = el.attrib.get("definitionURL", "").strip()
        if url == _TIME:
            return ("time",)
        if url == _AVOGADRO:
            return ("num", 6.02214076e23)
        raise UnsupportedSBML(f"csymbol {url or '(no definitionURL)'}")
    if tag in _CONSTANTS:
        return ("num", _CONSTANTS[tag])
    if tag == "piecewise":
        pieces, otherwise = [], None
        for c in el:
            kids = list(c)
            if _local(c.tag) == "piece" and len(kids) == 2:
                pieces.append((parse_mathml(kids[0]), parse_mathml(kids[1])))
            elif _local(c.tag) == "otherwise" and len(kids) == 1:
                otherwise = parse_mathml(kids[0])
            else:
                raise SBMLError("malformed <piecewise>")
        return ("piecewise", tuple(pieces), otherwise)
    if tag == "apply":
        kids = list(el)
        if not kids:
            raise SBMLError("empty <apply>")
        head, rest = kids[0], kids[1:]
        h = _local(head.tag)
        if h == "ci":
            return ("call", "".join(head.itertext()).strip(), tuple(parse_mathml(k) for k in rest))
        if h == "csymbol":
            raise UnsupportedSBML(f"csymbol function {head.attrib.get('definitionURL', '')}")
        if h not in _OPERATORS:
            raise UnsupportedSBML(f"MathML operator <{h}>")
        args = []
        for k in rest:
            q = _local(k.tag)
            if q in ("degree", "logbase"):
                inner = list(k)
                if len(inner) != 1:
                    raise SBMLError(f"<{q}> must contain one expression")
                args.insert(0, parse_mathml(inner[0]))
            else:
                args.append(parse_mathml(k))
        lo, hi = _ARITY.get(h, (1, 1) if h in _UNARY else (0, None))
        if h in ("root", "log"):
            lo, hi = 1, 2
        if len(args) < lo or (hi is not None and len(args) > hi):
            raise SBMLError(f"<{h}> applied to {len(args)} arguments")
        return ("op", h, tuple(args))
    raise UnsupportedSBML(f"MathML element <{tag}>")


def symbols(node) -> set[str]:
    kind = node[0]
    if kind == "sym":
        return {node[1]}
    if kind == "op" or kind == "call":
        out = set()
        for a in node[2]:
            out |= symbols(a)
        return out
    if kind == "piecewise":
        out = set()
        for v, c in node[1]:
            out |= symbols(v) | symbols(c)
        if node[2] is not None:
            out |= symbols(node[2])
        return out
    return set()


def infix(node) -> str:
    """Human-readable form for graph attributes and reports (not re-parsed)."""
    kind = node[0]
    if kind == "num":
        return repr(node[1]) if not float(node[1]).is_integer() else str(int(node[1]))
    if kind == "sym":
        return node[1]
    if kind == "time":
        return "time"
    if kind == "call":
        return f"{node[1]}({', '.join(infix(a) for a in node[2])})"
    if kind == "piecewise":
        body = ", ".join(f"{infix(v)} if {infix(c)}" for v, c in node[1])
        return f"piecewise({body}{', otherwise ' + infix(node[2]) if node[2] else ''})"
    name, args = node[1], node[2]
    if name in _INFIX and len(args) >= 2:
        return "(" + f" {_INFIX[name]} ".join(infix(a) for a in args) + ")"
    if name == "minus" and len(args) == 1:
        return f"-{infix(args[0])}"
    return f"{name}({', '.join(infix(a) for a in args)})"


def _substitute(node, mapping):
    kind = node[0]
    if kind == "sym":
        return mapping.get(node[1], node)
    if kind in ("op", "call"):
        return (kind, node[1], tuple(_substitute(a, mapping) for a in node[2]))
    if kind == "piecewise":
        return ("piecewise", tuple((_substitute(v, mapping), _substitute(c, mapping)) for v, c in node[1]),
                None if node[2] is None else _substitute(node[2], mapping))
    return node


def _inline(node, functions, depth=0):
    """Replace calls to SBML function definitions by their bodies."""
    if depth > 64:
        raise SBMLError("function definitions nest too deeply (recursive?)")
    kind = node[0]
    if kind == "call":
        if node[1] not in functions:
            raise SBMLError(f"call to undefined function {node[1]!r}")
        params, body = functions[node[1]]
        args = [_inline(a, functions, depth) for a in node[2]]
        if len(args) != len(params):
            raise SBMLError(f"function {node[1]!r} takes {len(params)} arguments, got {len(args)}")
        return _inline(_substitute(body, dict(zip(params, args))), functions, depth + 1)
    if kind == "op":
        return ("op", node[1], tuple(_inline(a, functions, depth) for a in node[2]))
    if kind == "piecewise":
        return ("piecewise", tuple((_inline(v, functions, depth), _inline(c, functions, depth))
                                   for v, c in node[1]),
                None if node[2] is None else _inline(node[2], functions, depth))
    return node


def compile_expr(node, index: dict[str, int], constants: dict[str, float] | None = None):
    """Closure ``f(env, t) -> float``; ``index`` maps symbols to ``env`` slots and
    ``constants`` binds local parameters (which shadow global symbols)."""
    constants = constants or {}
    kind = node[0]
    if kind == "num":
        v = float(node[1])
        return lambda env, t: v
    if kind == "time":
        return lambda env, t: t
    if kind == "sym":
        name = node[1]
        if name in constants:
            v = float(constants[name])
            return lambda env, t: v
        if name not in index:
            raise SBMLError(f"math references unknown or unsupported symbol {name!r}")
        i = index[name]
        return lambda env, t: env[i]
    if kind == "piecewise":
        pieces = [(compile_expr(v, index, constants), compile_expr(c, index, constants)) for v, c in node[1]]
        other = compile_expr(node[2], index, constants) if node[2] is not None else (lambda env, t: math.nan)

        def piecewise(env, t):
            for value, cond in pieces:
                if cond(env, t):
                    return value(env, t)
            return other(env, t)
        return piecewise
    if kind == "call":
        raise SBMLError("function calls must be inlined before compilation")
    fn = _OPERATORS[node[1]]
    args = [compile_expr(a, index, constants) for a in node[2]]
    if len(args) == 1:
        a0 = args[0]
        return lambda env, t: fn([a0(env, t)])
    if len(args) == 2:
        a0, a1 = args
        return lambda env, t: fn([a0(env, t), a1(env, t)])
    return lambda env, t: fn([a(env, t) for a in args])


# ---------------------------------------------------------------------------
# model components
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Compartment:
    id: str
    size: float | None
    name: str = ""


@dataclass(frozen=True)
class Species:
    id: str
    compartment: str
    initial: tuple[str, float] | None       # ("amount" | "concentration", value) as written
    has_only_substance_units: bool
    boundary_condition: bool
    constant: bool
    name: str = ""


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: tuple[tuple[str, Fraction], ...]
    products: tuple[tuple[str, Fraction], ...]
    modifiers: tuple[str, ...]
    rate: tuple | None                       # inlined expression tree, or None (no kinetic law)
    local_parameters: dict = field(default_factory=dict)
    reversible: bool = True
    name: str = ""


def _bool(el, attr, default, level):
    v = el.attrib.get(attr)
    if v is None:
        if level >= 3 and default is None:
            raise SBMLError(f"<{_local(el.tag)} id={el.attrib.get('id')!r}> needs attribute {attr!r} in Level 3")
        return bool(default)
    if v not in ("true", "false", "1", "0"):
        raise SBMLError(f"attribute {attr}={v!r} is not a boolean")
    return v in ("true", "1")


def _float(el, attr):
    v = el.attrib.get(attr)
    if v is None:
        return None
    x = float(v)
    if not math.isfinite(x):
        raise SBMLError(f"attribute {attr}={v!r} is not finite")
    return x


def _stoichiometry(ref, level) -> Fraction:
    for c in ref:
        if _local(c.tag) == "stoichiometryMath":
            raise UnsupportedSBML("stoichiometryMath")
    raw = ref.attrib.get("stoichiometry")
    if raw is None:
        if level >= 3:
            raise UnsupportedSBML(f"Level 3 speciesReference to {ref.attrib.get('species')!r} without stoichiometry")
        raw = "1"
    try:
        value = Fraction(raw.strip())
    except ValueError as exc:
        raise SBMLError(f"stoichiometry {raw!r} is not a finite decimal") from exc
    if value <= 0:
        raise SBMLError(f"stoichiometry {raw!r} must be positive")
    return value


_IGNORED = {"notes", "annotation", "listOfUnitDefinitions", "listOfCompartmentTypes", "listOfSpeciesTypes"}
_HANDLED = {"listOfFunctionDefinitions", "listOfCompartments", "listOfSpecies", "listOfParameters",
            "listOfInitialAssignments", "listOfRules", "listOfReactions"}


class SBMLModel:
    """An SBML reaction network: structure (exact stoichiometry), executable
    kinetics, and conversion to a typed :class:`Graph`."""

    def __init__(self, *, id, name, level, version, compartments, species, parameters,
                 functions, initial_assignments, rules, reactions, source=None):
        self.id, self.name, self.level, self.version = id, name, level, version
        self.compartments: dict[str, Compartment] = compartments
        self.species: dict[str, Species] = species
        self.parameters: dict[str, tuple[float | None, bool]] = parameters
        self.functions = functions
        self.initial_assignments: dict[str, tuple] = initial_assignments
        self.rules: dict[str, tuple] = rules
        self.reactions: dict[str, Reaction] = reactions
        self.source = source or {}
        self._check_semantics()
        self._compile()

    # --- structure -----------------------------------------------------------
    def _check_semantics(self):
        ids = list(self.compartments) + list(self.species) + list(self.parameters) + list(self.reactions)
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise SBMLError(f"duplicate ids {sorted(dup)}")
        for s in self.species.values():
            if s.compartment not in self.compartments:
                raise SBMLError(f"species {s.id!r} in unknown compartment {s.compartment!r}")
        for target in self.rules:
            if target in self.compartments:
                raise UnsupportedSBML(f"assignment rule for compartment {target!r} (variable volume)")
            if target not in self.species and target not in self.parameters:
                raise SBMLError(f"assignment rule for unknown or reaction id {target!r}")
            if target in self.initial_assignments:
                raise SBMLError(f"{target!r} has both an assignment rule and an initial assignment")
            if (target in self.parameters and self.parameters[target][1]) or (
                    target in self.species and self.species[target].constant):
                raise SBMLError(f"constant {target!r} cannot have an assignment rule")
        for target in self.initial_assignments:
            if target not in self.compartments and target not in self.species and target not in self.parameters:
                raise UnsupportedSBML(f"initial assignment to {target!r} (not a compartment, species or parameter)")
        for c in self.compartments.values():
            if c.size is None and c.id not in self.initial_assignments:
                raise SBMLError(f"compartment {c.id!r} has no size")
        for s in self.species.values():
            if s.initial is None and s.id not in self.initial_assignments and s.id not in self.rules:
                raise SBMLError(f"species {s.id!r} has no initial value")
        for p, (value, _) in self.parameters.items():
            if value is None and p not in self.initial_assignments and p not in self.rules:
                raise SBMLError(f"parameter {p!r} has no value")
        for r in self.reactions.values():
            for sid, _ in r.reactants + r.products:
                if sid not in self.species:
                    raise SBMLError(f"reaction {r.id!r} references unknown species {sid!r}")
                sp = self.species[sid]
                if not sp.boundary_condition and (sp.constant or sid in self.rules):
                    raise SBMLError(f"species {sid!r} is changed by reaction {r.id!r} but is "
                                    f"{'constant' if sp.constant else 'set by an assignment rule'} "
                                    "and not a boundary condition")
            for sid in r.modifiers:
                if sid not in self.species:
                    raise SBMLError(f"reaction {r.id!r} modifier {sid!r} is not a species")

    @property
    def dynamic_species(self) -> list[str]:
        """Species whose amounts are state variables (changed only by reactions)."""
        return [s.id for s in self.species.values()
                if not s.boundary_condition and not s.constant and s.id not in self.rules]

    @property
    def reaction_ids(self) -> list[str]:
        return list(self.reactions)

    def stoichiometry_matrix(self, species_ids: list[str] | None = None) -> list[list[Fraction]]:
        """Exact net stoichiometry N (rows: species, columns: reactions)."""
        species_ids = self.dynamic_species if species_ids is None else species_ids
        row = {s: i for i, s in enumerate(species_ids)}
        n = [[Fraction(0)] * len(self.reactions) for _ in species_ids]
        for j, r in enumerate(self.reactions.values()):
            for sid, c in r.reactants:
                if sid in row:
                    n[row[sid]][j] -= c
            for sid, c in r.products:
                if sid in row:
                    n[row[sid]][j] += c
        return n

    def conservation_laws(self) -> list[dict[str, int]]:
        """Exact conserved moieties over dynamic species amounts (see stoichiometry)."""
        ids = self.dynamic_species
        laws = S.conservation_laws(self.stoichiometry_matrix(ids))
        return [{ids[i]: c for i, c in enumerate(law) if c} for law in laws]

    def reacting_species(self) -> list[str]:
        """Every species consumed or produced by some reaction, in document order."""
        used = {sid for r in self.reactions.values() for sid, _ in r.reactants + r.products}
        return [s for s in self.species if s in used]

    def stoichiometric_consistency(self) -> dict:
        """Positive-mass check over all reacting species.

        Boundary and constant species keep their rows (being held fixed does
        not change a species' mass), but a reaction counts as internal only if
        each side has at least one species that is not held fixed. Reactions
        whose reactants or products are all fixed pools (for example
        ``A -> EmptySet``, or synthesis from a clamped precursor) exchange
        matter with the environment, like pure synthesis or degradation."""
        ids = self.reacting_species()
        free = {s.id for s in self.species.values() if not s.boundary_condition and not s.constant}
        internal = [any(sid in free for sid, _ in r.reactants) and any(sid in free for sid, _ in r.products)
                    for r in self.reactions.values()]
        res = S.stoichiometric_consistency(self.stoichiometry_matrix(ids), internal)
        return {**res,
                "conserved": [ids[i] for i in res["conserved"]],
                "unconserved": [ids[i] for i in res["unconserved"]],
                "internal_reactions": [self.reaction_ids[j] for j in res["internal_reactions"]],
                "mass": None if res["mass"] is None else {ids[i]: str(m) for i, m in enumerate(res["mass"])},
                "leak_combination": None if res["leak_combination"] is None else {
                    self.reaction_ids[j]: str(v) for j, v in zip(res["internal_reactions"], res["leak_combination"])
                    if v}}

    # --- kinetics ------------------------------------------------------------
    def _compile(self):
        self._symbols = list(self.compartments) + list(self.species) + list(self.parameters)
        self._index = {s: i for i, s in enumerate(self._symbols)}
        allowed = set(self._index)

        def check_refs(expr, where, local=()):
            unknown = symbols(expr) - allowed - set(local)
            if unknown & set(self.reactions):
                raise UnsupportedSBML(f"{where} uses a reaction id as a rate symbol")
            if unknown:
                raise SBMLError(f"{where} references unknown symbols {sorted(unknown)}")

        for target, expr in {**self.rules, **self.initial_assignments}.items():
            check_refs(expr, f"rule/assignment for {target!r}")
        self._rule_order = _topological(self.rules, set(self.rules))
        self._rule_fns = [(self._index[t], compile_expr(self.rules[t], self._index)) for t in self._rule_order]
        self._rate_fns = []
        for r in self.reactions.values():
            if r.rate is None:
                self._rate_fns.append(None)
                continue
            check_refs(r.rate, f"kinetic law of {r.id!r}", r.local_parameters)
            self._rate_fns.append(compile_expr(r.rate, self._index, r.local_parameters))
        self._missing_laws = [r for r, fn in zip(self.reactions, self._rate_fns) if fn is None]
        dyn = self.dynamic_species
        self._dyn_env = np.array([self._index[s] for s in dyn], dtype=int)
        self._dyn_size_ix = np.array([self._index[self.species[s].compartment] for s in dyn], dtype=int)
        self._dyn_conc = np.array([not self.species[s].has_only_substance_units for s in dyn], dtype=bool)
        n = self.stoichiometry_matrix(dyn)
        self._n = np.array([[float(x) for x in row] for row in n]).reshape(len(dyn), len(self.reactions))
        self._env0 = self._initial_env()

    def _initial_env(self) -> np.ndarray:
        """Symbol values at t = 0: attributes, then initial assignments and rules in
        dependency order. Species values are concentrations unless
        hasOnlySubstanceUnits."""
        env = np.full(len(self._symbols), np.nan)
        defs: dict[str, tuple] = {}
        for c in self.compartments.values():
            if c.size is not None:
                env[self._index[c.id]] = c.size
        for p, (value, _) in self.parameters.items():
            if value is not None:
                env[self._index[p]] = value
        raw_species = {}
        for s in self.species.values():
            if s.id in self.initial_assignments or s.id in self.rules:
                continue
            raw_species[s.id] = s
            defs[s.id] = ("sym", s.compartment)          # dependency marker only
        defs.update(self.initial_assignments)
        defs.update(self.rules)
        order = _topological(defs, set(defs))
        for target in order:
            i = self._index[target]
            if target in raw_species:
                s = raw_species[target]
                kind, value = s.initial
                size = env[self._index[s.compartment]]
                as_amount = value if kind == "amount" else value * size
                env[i] = as_amount if s.has_only_substance_units else _ieee_div(as_amount, size)
            else:
                env[i] = compile_expr(_inline(defs[target], self.functions), self._index)(env, 0.0)
        if np.isnan(env).any():
            missing = [s for s, v in zip(self._symbols, env) if np.isnan(v)]
            raise SBMLError(f"no initial value for {missing}")
        return env

    def sizes(self) -> np.ndarray:
        """Compartment size of each dynamic species (state scaling)."""
        return self._env0[self._dyn_size_ix].copy()

    def initial_amounts(self) -> np.ndarray:
        """Initial amount of each dynamic species, in model substance units."""
        v = self._env0[self._dyn_env]
        return np.where(self._dyn_conc, v * self.sizes(), v)

    def _env(self, amounts, t) -> np.ndarray:
        env = self._env0.copy()
        sizes = env[self._dyn_size_ix]
        env[self._dyn_env] = np.where(self._dyn_conc, amounts / sizes, amounts)
        for i, fn in self._rule_fns:
            env[i] = fn(env, t)
        return env

    def rates(self, amounts, t: float = 0.0) -> np.ndarray:
        """Reaction rates (substance / time) at the given dynamic species amounts."""
        if self._missing_laws:
            raise SBMLError(f"reactions without kinetic laws cannot be simulated: {self._missing_laws}")
        env = self._env(np.asarray(amounts, dtype=float), t)
        return np.array([fn(env, t) for fn in self._rate_fns], dtype=float)

    def amount_derivative(self, amounts, t: float = 0.0) -> np.ndarray:
        return self._n @ self.rates(amounts, t) if len(self.reactions) else np.zeros(len(amounts))

    def simulate(self, times, rtol=1e-10, atol=1e-12, method="LSODA", amounts0=None) -> dict:
        """Integrate from t = times[0]. The integrated state is amount / size
        (concentration-scaled) so tolerances are comparable across compartments."""
        from scipy.integrate import solve_ivp

        times = np.asarray(times, dtype=float)
        sizes = self.sizes()
        y0 = (self.initial_amounts() if amounts0 is None else np.asarray(amounts0, float)) / sizes
        sol = solve_ivp(lambda t, y: self.amount_derivative(y * sizes, t) / sizes,
                        (times[0], times[-1]), y0, t_eval=times, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        amounts = (sol.y.T * sizes)
        conc = np.where(self._dyn_conc, amounts / sizes, amounts)
        return {"time": sol.t, "species": self.dynamic_species, "amount": amounts,
                "value": conc, "nfev": int(sol.nfev)}

    def _reduction(self):
        """Exact conservation laws in RREF on amounts: x_P = T - M x_Q."""
        ids = self.dynamic_species
        laws = S.conservation_laws(self.stoichiometry_matrix(ids))
        if not laws:
            return [], [], list(range(len(ids))), np.zeros((0, len(ids)))
        r, pivots = S.rref(laws)
        free = [c for c in range(len(ids)) if c not in pivots]
        return r, pivots, free, np.array([[float(x) for x in row] for row in r])

    def jacobian(self, amounts, t: float = 0.0, rel_step=1e-7) -> np.ndarray:
        """d(amount derivative)/d(amount) by central differences."""
        x = np.asarray(amounts, dtype=float)
        scale = np.maximum(np.abs(x), self.sizes() * 1e-9)
        j = np.empty((len(x), len(x)))
        for k in range(len(x)):
            h = rel_step * scale[k]
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            j[:, k] = (self.amount_derivative(xp, t) - self.amount_derivative(xm, t)) / (2 * h)
        return j

    def steady_state(self, amounts_guess=None, tol=1e-13) -> dict:
        """Newton/hybrid solve of N v(x) = 0 on the conservation-reduced system.

        The conserved totals are fixed at those of ``amounts_guess`` (default:
        the initial amounts), so the answer is the steady state of that
        stoichiometric class, nearest the guess when several exist.
        """
        from scipy.optimize import root

        x0 = self.initial_amounts() if amounts_guess is None else np.asarray(amounts_guess, float)
        sizes = self.sizes()
        _, pivots, free, lr = self._reduction()
        totals = lr @ x0 if len(pivots) else np.zeros(0)

        def residual(y):
            x = y * sizes
            f = self.amount_derivative(x) / sizes
            return np.concatenate([f[free], (lr @ x - totals) / np.maximum(np.abs(totals), 1e-300)])

        sol = root(residual, x0 / sizes, method="hybr", tol=tol)
        x = sol.x * sizes
        f = self.amount_derivative(x)
        return {"amount": x, "species": self.dynamic_species, "success": bool(sol.success),
                "max_abs_derivative": float(np.max(np.abs(f / sizes))) if len(f) else 0.0,
                "conservation_residual": float(np.max(np.abs(lr @ x - totals))) if len(pivots) else 0.0}

    def reduced_eigenvalues(self, amounts) -> np.ndarray:
        """Eigenvalues of the Jacobian restricted to the stoichiometric class
        (the conservation laws' zero modes removed via the link matrix)."""
        j = self.jacobian(amounts)
        _, pivots, free, lr = self._reduction()
        link = np.zeros((len(amounts), len(free)))
        for col, q in enumerate(free):
            link[q, col] = 1.0
        for row, p in enumerate(pivots):
            link[p, :] = -lr[row, free]
        return np.linalg.eigvals(j[np.ix_(free, range(len(amounts)))] @ link)

    # --- graph ---------------------------------------------------------------
    def to_graph(self) -> tuple[Graph, Schema]:
        """Typed graph: compartments, species, reactions and exact conserved
        moieties, with the schema and invariants in :func:`sbml_schema`."""
        g = Graph(self.id or "sbml")
        env0 = self._env0
        for c in self.compartments.values():
            g.add_node(c.id, "compartment", "cell", c.name or c.id, size=float(env0[self._index[c.id]]))
        x0 = dict(zip(self.dynamic_species, self.initial_amounts()))
        for s in self.species.values():
            v = env0[self._index[s.id]]
            amount = v if s.has_only_substance_units else v * env0[self._index[s.compartment]]
            g.add_node(s.id, "species", "molecule", s.name or s.id,
                       compartment=s.compartment, initial_amount=float(x0.get(s.id, amount)),
                       boundary_condition=s.boundary_condition, constant=s.constant,
                       has_only_substance_units=s.has_only_substance_units,
                       assignment_rule=infix(self.rules[s.id]) if s.id in self.rules else None)
            g.add_edge(s.id, s.compartment, "located_in")
        for r in self.reactions.values():
            g.add_node(r.id, "reaction", "cell", r.name or r.id, reversible=r.reversible,
                       kinetic_law=None if r.rate is None else infix(r.rate),
                       local_parameters={k: float(v) for k, v in r.local_parameters.items()})
            key = f"sbml:{self.id}/{r.id}"
            for sid, c in r.reactants:
                g.add_edge(sid, r.id, "reactant", model=key, stoichiometry=str(c))
            for sid, c in r.products:
                g.add_edge(r.id, sid, "product", model=key, stoichiometry=str(c))
            for sid in r.modifiers:
                g.add_edge(sid, r.id, "modifier", model=key)
        for k, law in enumerate(self.conservation_laws()):
            total = sum(c * x0[s] for s, c in law.items())
            mid = f"moiety_{k + 1}"
            g.add_node(mid, "moiety", "molecule", " + ".join(
                (f"{c}*{s}" if c != 1 else s) for s, c in law.items()), total=float(total))
            for s, c in law.items():
                g.add_edge(s, mid, "member", coefficient=str(c))
        return g, sbml_schema()


def _topological(defs: dict[str, tuple], targets: set[str]) -> list[str]:
    """Order definitions so each follows the targets it references."""
    deps = {t: symbols(e) & targets - {t} for t, e in defs.items()}
    for t, e in defs.items():
        if t in symbols(e) and e[0] != "sym":
            raise SBMLError(f"{t!r} is defined in terms of itself")
    order, done, active = [], set(), set()

    def visit(t):
        if t in done:
            return
        if t in active:
            raise SBMLError(f"algebraic loop through {t!r}")
        active.add(t)
        for d in sorted(deps[t]):
            visit(d)
        active.discard(t)
        done.add(t)
        order.append(t)

    for t in defs:
        visit(t)
    return order


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _refuse_dtd(data: bytes) -> None:
    """Refuse any DOCTYPE or entity declaration before the real parse.

    Expat decodes the document in whatever encoding it declares (UTF-8,
    UTF-16 with or without a byte-order mark, Latin-1, ...), so the check
    sees the declarations themselves. A scan of the raw bytes would not: in
    UTF-16, ``<!DOCTYPE`` is interleaved with NUL bytes. Encodings expat cannot
    read are refused as malformed rather than passed through unchecked.
    """
    import xml.parsers.expat as expat

    def refuse(*_):
        raise SBMLError("documents with a DOCTYPE or entity declarations are refused")

    parser = expat.ParserCreate()
    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    try:
        parser.Parse(data, True)
    except expat.ExpatError as exc:
        raise SBMLError(f"malformed XML: {exc}") from exc


def read_sbml(source) -> SBMLModel:
    """Read an SBML document from a path or bytes. A path's SHA-256 is recorded."""
    if isinstance(source, (bytes, bytearray)):
        data, origin = bytes(source), None
    else:
        origin = Path(source)
        data = origin.read_bytes()
    _refuse_dtd(data)
    root = ET.fromstring(data)
    if _local(root.tag) != "sbml":
        raise SBMLError("root element is not <sbml>")
    ns = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
    level, version = int(root.attrib.get("level", 0)), int(root.attrib.get("version", 0))
    if (level, version) not in {(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2)}:
        raise UnsupportedSBML(f"SBML Level {level} Version {version}")
    for attr, value in root.attrib.items():
        if _local(attr) == "required" and value == "true":
            raise UnsupportedSBML(f"required SBML package ({attr})")

    def q(tag):
        return f"{{{ns}}}{tag}" if ns else tag

    model = root.find(q("model"))
    if model is None:
        raise SBMLError("no <model>")
    if level >= 3 and "conversionFactor" in model.attrib:
        raise UnsupportedSBML("model conversionFactor")
    for child in model:
        name = _local(child.tag)
        if name in _IGNORED or (name in _HANDLED and child.tag == q(name)):
            continue
        if len(child) == 0 and name.startswith("listOf"):
            continue
        raise UnsupportedSBML(f"model element <{name}>")

    def items(list_tag, item_tag):
        lst = model.find(q(list_tag))
        if lst is None:
            return []
        out = []
        for el in lst:
            if _local(el.tag) in ("notes", "annotation"):
                continue
            if el.tag != q(item_tag):
                raise UnsupportedSBML(f"<{_local(el.tag)}> in <{list_tag}>")
            out.append(el)
        return out

    def math_of(el, where):
        m = el.find(f"{{{MATHML}}}math")
        if m is None:
            raise SBMLError(f"{where} has no MathML")
        return parse_mathml(m)

    functions = {}
    for fd in items("listOfFunctionDefinitions", "functionDefinition"):
        m = fd.find(f"{{{MATHML}}}math")
        if m is None:
            raise SBMLError(f"function {fd.attrib.get('id')!r} has no MathML")
        lam_el = [c for c in m if _local(c.tag) not in ("annotation", "annotation-xml")]
        if len(lam_el) != 1 or _local(lam_el[0].tag) != "lambda":
            raise SBMLError(f"function {fd.attrib.get('id')!r} is not a lambda")
        kids = list(lam_el[0])
        params = []
        for k in kids[:-1]:
            if _local(k.tag) != "bvar" or len(k) != 1:
                raise SBMLError(f"function {fd.attrib.get('id')!r}: malformed <bvar>")
            params.append("".join(k[0].itertext()).strip())
        functions[fd.attrib["id"]] = (tuple(params), parse_mathml(kids[-1]))

    compartments = {}
    for c in items("listOfCompartments", "compartment"):
        if not _bool(c, "constant", True if level < 3 else None, level):
            raise UnsupportedSBML(f"non-constant compartment {c.attrib['id']!r}")
        size = _float(c, "size")
        compartments[c.attrib["id"]] = Compartment(c.attrib["id"], size, c.attrib.get("name", ""))

    species = {}
    for s in items("listOfSpecies", "species"):
        if "conversionFactor" in s.attrib:
            raise UnsupportedSBML(f"species conversionFactor on {s.attrib['id']!r}")
        amount, conc = _float(s, "initialAmount"), _float(s, "initialConcentration")
        if amount is not None and conc is not None:
            raise SBMLError(f"species {s.attrib['id']!r} has both initialAmount and initialConcentration")
        initial = ("amount", amount) if amount is not None else ("concentration", conc) if conc is not None else None
        species[s.attrib["id"]] = Species(
            s.attrib["id"], s.attrib.get("compartment", ""), initial,
            _bool(s, "hasOnlySubstanceUnits", False if level < 3 else None, level),
            _bool(s, "boundaryCondition", False if level < 3 else None, level),
            _bool(s, "constant", False if level < 3 else None, level),
            s.attrib.get("name", ""))

    parameters = {}
    for p in items("listOfParameters", "parameter"):
        parameters[p.attrib["id"]] = (_float(p, "value"), _bool(p, "constant", True if level < 3 else None, level))

    initial_assignments = {}
    for ia in items("listOfInitialAssignments", "initialAssignment"):
        initial_assignments[ia.attrib["symbol"]] = _inline(math_of(ia, "initialAssignment"), functions)

    rules = {}
    lst = model.find(q("listOfRules"))
    for rule in (lst if lst is not None else []):
        kind = _local(rule.tag)
        if kind in ("notes", "annotation"):
            continue
        if kind != "assignmentRule":
            raise UnsupportedSBML(f"<{kind}>")
        var = rule.attrib["variable"]
        if var in rules:
            raise SBMLError(f"two assignment rules for {var!r}")
        rules[var] = _inline(math_of(rule, f"rule for {var!r}"), functions)

    reactions = {}
    for r in items("listOfReactions", "reaction"):
        rid = r.attrib["id"]
        if _bool(r, "fast", False if (level < 3 or version >= 2) else None, level):
            raise UnsupportedSBML(f"fast reaction {rid!r}")
        def refs(tag, item):
            lst = r.find(q(tag))
            out = []
            for el in (lst if lst is not None else []):
                if _local(el.tag) in ("notes", "annotation"):
                    continue
                if el.tag != q(item):
                    raise UnsupportedSBML(f"<{_local(el.tag)}> in <{tag}>")
                out.append(el)
            return out

        reactants = tuple((x.attrib["species"], _stoichiometry(x, level))
                          for x in refs("listOfReactants", "speciesReference"))
        products = tuple((x.attrib["species"], _stoichiometry(x, level))
                         for x in refs("listOfProducts", "speciesReference"))
        modifiers = tuple(x.attrib["species"] for x in refs("listOfModifiers", "modifierSpeciesReference"))
        kl = r.find(q("kineticLaw"))
        rate, local = None, {}
        if kl is not None:
            for tag, item in (("listOfParameters", "parameter"), ("listOfLocalParameters", "localParameter")):
                lst = kl.find(q(tag))
                for p in (lst if lst is not None else []):
                    if _local(p.tag) in ("notes", "annotation"):
                        continue
                    if p.tag != q(item):
                        raise UnsupportedSBML(f"<{_local(p.tag)}> in kinetic law of {rid!r}")
                    v = _float(p, "value")
                    if v is None:
                        raise SBMLError(f"local parameter {p.attrib['id']!r} of {rid!r} has no value")
                    local[p.attrib["id"]] = v
            rate = _inline(math_of(kl, f"kinetic law of {rid!r}"), functions)
        reactions[rid] = Reaction(rid, reactants, products, modifiers, rate, local,
                                  _bool(r, "reversible", True if level < 3 else None, level),
                                  r.attrib.get("name", ""))

    src = {"sha256": hashlib.sha256(data).hexdigest()}
    if origin is not None:
        src["path"] = str(origin)
    return SBMLModel(id=model.attrib.get("id", ""), name=model.attrib.get("name", ""),
                     level=level, version=version, compartments=compartments, species=species,
                     parameters=parameters, functions=functions,
                     initial_assignments=initial_assignments, rules=rules,
                     reactions=reactions, source=src)


# ---------------------------------------------------------------------------
# schema and invariants for an SBML-derived graph
# ---------------------------------------------------------------------------
def _graph_stoichiometry(g: Graph, dynamic_only: bool = True):
    """N recomputed from graph edges alone (independent of SBMLModel): rows are
    the dynamic species, or with ``dynamic_only=False`` every reacting species."""
    if dynamic_only:
        dyn = [n.id for n in g.nodes_of_type("species")
               if not n.attrs.get("boundary_condition") and not n.attrs.get("constant")
               and n.attrs.get("assignment_rule") is None]
    else:
        used = {e.src for e in g.edges_of_type("reactant")} | {e.dst for e in g.edges_of_type("product")}
        dyn = [n.id for n in g.nodes_of_type("species") if n.id in used]
    rx = [n.id for n in g.nodes_of_type("reaction")]
    row, col = {s: i for i, s in enumerate(dyn)}, {r: j for j, r in enumerate(rx)}
    n = [[Fraction(0)] * len(rx) for _ in dyn]
    for e in g.edges_of_type("reactant"):
        if e.src in row:
            n[row[e.src]][col[e.dst]] -= Fraction(e.attrs["stoichiometry"])
    for e in g.edges_of_type("product"):
        if e.dst in row:
            n[row[e.dst]][col[e.src]] += Fraction(e.attrs["stoichiometry"])
    return dyn, rx, n


def sbml_schema() -> Schema:
    schema = Schema(
        node_types={"compartment": {"size"}, "species": {"compartment", "initial_amount", "boundary_condition"},
                    "reaction": {"reversible"}, "moiety": {"total"}},
        edge_rules=[
            EdgeRule("reactant", {"species"}, {"reaction"}, {"stoichiometry"}),
            EdgeRule("product", {"reaction"}, {"species"}, {"stoichiometry"}),
            EdgeRule("modifier", {"species"}, {"reaction"}),
            EdgeRule("located_in", {"species"}, {"compartment"}),
            EdgeRule("member", {"species"}, {"moiety"}, {"coefficient"}),
        ])

    @schema.invariant("stoichiometry")
    def _stoich(g: Graph) -> list[str]:
        out = []
        for e in g.edges_of_type("reactant", "product"):
            try:
                c = Fraction(str(e.attrs.get("stoichiometry")))
                ok = c > 0
            except (TypeError, ValueError, ZeroDivisionError):
                ok = False
            if not ok:
                out.append(f"edge {e.key}: stoichiometry must be a positive exact rational")
        return out

    @schema.invariant("located")
    def _located(g: Graph) -> list[str]:
        return [f"species {n.id}: needs exactly one compartment"
                for n in g.nodes_of_type("species") if len(g.out_edges(n.id, ("located_in",))) != 1]

    @schema.invariant("conservation")
    def _conservation(g: Graph) -> list[str]:
        """Each moiety node must be an exact left-null vector of N, the moieties
        must be independent and span the whole left nullspace, and each total
        must equal the initial amounts it weights."""
        try:
            dyn, _, n = _graph_stoichiometry(g)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            return [f"cannot rebuild stoichiometry: {exc}"]
        out, laws = [], []
        idx = {s: i for i, s in enumerate(dyn)}
        for m in g.nodes_of_type("moiety"):
            law = [Fraction(0)] * len(dyn)
            for e in g.in_edges(m.id, ("member",)):
                if e.src not in idx:
                    out.append(f"{m.id}: member {e.src} is not a dynamic species")
                    continue
                law[idx[e.src]] = Fraction(e.attrs["coefficient"])
            if S.conservation_residual([law], n) != 0:
                out.append(f"{m.id}: not conserved by every reaction (l.N != 0)")
            total = sum(float(c) * g.nodes[s].attrs["initial_amount"] for s, c in zip(dyn, law) if c)
            if not math.isclose(total, m.attrs["total"], rel_tol=1e-12, abs_tol=1e-300):
                out.append(f"{m.id}: total {m.attrs['total']} != weighted initial amounts {total}")
            laws.append(law)
        expected = len(dyn) - S.rank(n) if n and n[0] else len(dyn)
        if laws and S.rank(laws) != len(laws):
            out.append("moieties are linearly dependent")
        if len(laws) != expected:
            out.append(f"{len(laws)} moieties recorded, but the stoichiometry has {expected} "
                       "independent conservation laws")
        return out

    @schema.invariant("stoichiometric-consistency")
    def _consistency(g: Graph) -> list[str]:
        """Every dynamic species must admit a positive mass conserved by all
        internal reactions (Gevorgyan et al. 2008)."""
        dyn, rx, n = _graph_stoichiometry(g, dynamic_only=False)
        fixed = {v.id for v in g.nodes_of_type("species")
                 if v.attrs.get("boundary_condition") or v.attrs.get("constant")}
        internal = [any(e.src not in fixed for e in g.in_edges(r, ("reactant",)))
                    and any(e.dst not in fixed for e in g.out_edges(r, ("product",))) for r in rx]
        res = S.stoichiometric_consistency(n, internal)
        out = []
        if res["unconserved"]:
            names = [dyn[i] for i in res["unconserved"]]
            leak = res["leak_combination"]
            how = ("exact certificate: net production by " + ", ".join(
                f"{v}*{rx[j]}" for j, v in zip(res["internal_reactions"], leak) if v)
                   if leak else "LP only; no exact certificate")
            out.append(f"species {names} cannot carry a positive conserved mass ({how})")
        elif not res["certified"]:
            out.append("consistency found by LP but no exact mass vector could be certified")
        return out

    return schema
