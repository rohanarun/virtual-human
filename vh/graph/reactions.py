"""Elementary mass-action networks in one fixed, well-mixed compartment.

Concentrations are mol/L, time is seconds. This is a numerical research primitive,
not a biological model. See docs/reaction_experiments.md for assumptions/limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .core import Graph


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _sources(values):
    if isinstance(values, str):
        raise ValueError("provenance must contain source identifiers")
    sources = tuple(_label(v, "provenance") for v in values)
    if not sources:
        raise ValueError("provenance must contain source identifiers")
    return sources


def rate_unit(order: int) -> str:
    """Canonical unit of an elementary rate constant of integer reaction order."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    return {0: "mol/L/s", 1: "1/s"}.get(order, f"(mol/L)^{1-order}/s")


@dataclass(frozen=True)
class Species:
    id: str
    provenance: tuple[str, ...]

    def __post_init__(self):
        _label(self.id, "species id")
        object.__setattr__(self, "provenance", _sources(self.provenance))


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    rate: float
    unit: str
    provenance: tuple[str, ...]

    def __post_init__(self):
        _label(self.id, "reaction id")
        for name in ("reactants", "products"):
            side = dict(getattr(self, name))
            for species, coefficient in side.items():
                _label(species, "species id")
                if isinstance(coefficient, bool) or not isinstance(coefficient, int) or coefficient <= 0:
                    raise ValueError("stoichiometric coefficients must be positive integers")
            object.__setattr__(self, name, MappingProxyType(side))
        if not self.reactants and not self.products:
            raise ValueError("a reaction must involve a species")
        object.__setattr__(self, "rate", _number(self.rate, "rate"))
        if self.unit != rate_unit(sum(self.reactants.values())):
            raise ValueError("rate unit does not match elementary reaction order")
        object.__setattr__(self, "provenance", _sources(self.provenance))


def _nullspace(rows, columns):
    """Exact rational nullspace by row reduction; rows are S transpose."""
    a = [[Fraction(x) for x in row] for row in rows]
    pivots = []
    row = 0
    for col in range(columns):
        pivot = next((i for i in range(row, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        divisor = a[row][col]
        a[row] = [x / divisor for x in a[row]]
        for i in range(len(a)):
            if i != row and a[i][col]:
                multiplier = a[i][col]
                a[i] = [x - multiplier * y for x, y in zip(a[i], a[row])]
        pivots.append(col)
        row += 1
    basis = []
    for free in sorted(set(range(columns)) - set(pivots)):
        vector = [Fraction(0) for _ in range(columns)]
        vector[free] = Fraction(1)
        for i, pivot in enumerate(pivots):
            vector[pivot] = -a[i][free]
        basis.append(vector)
    return basis


@dataclass
class Trajectory:
    times_s: np.ndarray
    species: tuple[str, ...]
    concentrations: np.ndarray
    sensitivity_reactions: tuple[str, ...]
    sensitivities: np.ndarray

    @property
    def final(self):
        return dict(zip(self.species, self.concentrations[-1].tolist()))


class ReactionNetwork:
    def __init__(self, species, reactions, *, context, provenance):
        self.species = tuple(species)
        self.reactions = tuple(reactions)
        if not self.species or not all(isinstance(s, Species) for s in self.species):
            raise ValueError("provide at least one Species")
        if not all(isinstance(r, Reaction) for r in self.reactions):
            raise ValueError("reactions must be Reaction objects")
        self.ids = tuple(s.id for s in self.species)
        self.reaction_ids = tuple(r.id for r in self.reactions)
        if len(set(self.ids)) != len(self.ids) or len(set(self.reaction_ids)) != len(self.reaction_ids):
            raise ValueError("species and reaction ids must be unique within their types")
        self.context = MappingProxyType(dict(context))
        if not self.context:
            raise ValueError("context must describe the modeled compartment/system")
        for key, value in self.context.items():
            _label(key, "context key")
            _label(value, "context value")
        self.provenance = _sources(provenance)
        for r in self.reactions:
            if (set(r.reactants) | set(r.products)) - set(self.ids):
                raise ValueError(f"reaction {r.id!r} references unknown species")
        self._orders = np.array([[r.reactants.get(s, 0) for s in self.ids]
                                 for r in self.reactions], dtype=int).reshape(len(self.reactions), len(self.ids))
        self._stoich_exact = tuple(tuple(r.products.get(s, 0) - r.reactants.get(s, 0)
                                         for r in self.reactions) for s in self.ids)
        self._stoich = np.array(self._stoich_exact, dtype=float)

    def conservation_basis(self):
        """Basis of w.T S = 0, exactly. Signed combinations are not chemical mass."""
        rows = list(zip(*self._stoich_exact))
        return tuple({s: w for s, w in zip(self.ids, vector) if w}
                     for vector in _nullspace(rows, len(self.ids)))

    def pool_residuals(self, weights):
        """Exact per-reaction residual of a declared pool (zero means conserved).

        Weights must be int/Fraction, avoiding float-to-rational ambiguity.
        """
        if not weights or set(weights) - set(self.ids):
            raise ValueError("pool must name known species")
        if any(isinstance(w, bool) or not isinstance(w, (int, Fraction)) for w in weights.values()):
            raise ValueError("pool weights must be integers or Fractions")
        return {r: sum((Fraction(weights.get(s, 0)) * self._stoich_exact[i][j]
                        for i, s in enumerate(self.ids)), Fraction(0))
                for j, r in enumerate(self.reaction_ids)}

    def _exact_effective_rates(self, rate_scales, exact_rates):
        """Exact rationals of the effective rates the integrator would use."""
        scales = {} if rate_scales is None else dict(rate_scales)
        exact = {} if exact_rates is None else dict(exact_rates)
        self._rates(scales)  # same validation as simulate
        if set(exact) - set(self.reaction_ids):
            raise ValueError("exact rates name an unknown reaction")
        values = []
        for r in self.reactions:
            if r.id in exact:
                k = exact[r.id]
                if isinstance(k, bool) or not isinstance(k, (int, Fraction)) or k < 0 or float(k) != r.rate:
                    raise ValueError(f"exact rate for {r.id!r} must be a rational whose float is the rate")
                k = Fraction(k)
            else:
                k = Fraction(r.rate)  # exact value of the stored binary float
            scale = scales.get(r.id, 1)
            values.append(k * (scale if isinstance(scale, Fraction) else Fraction(scale)))
        return values

    def _monomial_columns(self, rates):
        """Rate-weighted stoichiometry summed over reactions sharing a reactant monomial."""
        groups = {}
        for j, r in enumerate(self.reactions):
            groups.setdefault(tuple(sorted(r.reactants.items())), []).append(j)
        columns = {}
        for key, members in groups.items():
            column = [sum((rates[j] * self._stoich_exact[i][j] for j in members), Fraction(0))
                      for i in range(len(self.ids))]
            label = "*".join(s if n == 1 else f"{s}^{n}" for s, n in key) or "1"
            columns[label] = column
        return columns

    def kinetic_invariant_basis(self, rate_scales=None, exact_rates=None):
        """Exact basis of w with w . dC/dt = 0 for every state at these effective rates.

        Mass-action dC/dt is a sum, over distinct reactant monomials m, of m(C)
        times the rate-weighted stoichiometry of the reactions sharing m. Distinct
        monomials are independent functions, so w is invariant iff it annihilates
        every such column. The span contains every stoichiometric pool; extra
        vectors depend on rate values, e.g. an exposure (AUC) or clearance identity.
        Rates are the exact values of the stored floats unless ``exact_rates``
        supplies rationals whose float equals the stored rate.
        """
        columns = self._monomial_columns(self._exact_effective_rates(rate_scales, exact_rates))
        return tuple({s: w for s, w in zip(self.ids, vector) if w}
                     for vector in _nullspace(list(columns.values()), len(self.ids)))

    def kinetic_residuals(self, weights, rate_scales=None, exact_rates=None):
        """Exact residual of a declared linear invariant per reactant monomial.

        All zeros means ``sum(w_i C_i)`` is constant along every trajectory of
        these effective rates. External retain/add events change it by their
        weighted jump; check those separately.
        """
        if not weights or set(weights) - set(self.ids):
            raise ValueError("invariant must name known species")
        if any(isinstance(w, bool) or not isinstance(w, (int, Fraction)) for w in weights.values()):
            raise ValueError("invariant weights must be integers or Fractions")
        columns = self._monomial_columns(self._exact_effective_rates(rate_scales, exact_rates))
        return {label: sum((Fraction(weights.get(s, 0)) * column[i] for i, s in enumerate(self.ids)), Fraction(0))
                for label, column in columns.items()}

    def to_graph(self):
        """Bipartite topology preserves multi-reactant stoichiometry and catalysts."""
        graph = Graph("reaction network")
        for s in self.species:
            graph.add_node(f"species:{s.id}", "chemical_species", "molecule", species_id=s.id,
                           context=dict(self.context), provenance=list(s.provenance),
                           network_provenance=list(self.provenance))
        for r in self.reactions:
            rid = f"reaction:{r.id}"
            graph.add_node(rid, "elementary_reaction", "molecule", reaction_id=r.id,
                           rate=r.rate, unit=r.unit, provenance=list(r.provenance),
                           context=dict(self.context), network_provenance=list(self.provenance))
            for s, n in r.reactants.items():
                graph.add_edge(f"species:{s}", rid, "reactant", stoichiometry=n)
            for s, n in r.products.items():
                graph.add_edge(rid, f"species:{s}", "product", stoichiometry=n)
        return graph

    def _state(self, initial):
        if set(initial) != set(self.ids):
            raise ValueError("initial concentrations must name every species exactly")
        return np.array([_number(initial[s], f"initial {s}") for s in self.ids])

    def _rates(self, scales):
        if set(scales) - set(self.reaction_ids):
            raise ValueError("rate scales name an unknown reaction")
        return np.array([_number(r.rate * _number(scales.get(r.id, 1), f"rate scale {r.id}"),
                                 f"effective rate {r.id}") for r in self.reactions])

    def simulate(self, initial, times_s, *, max_step_s, rate_scales=None, sensitivity_reactions=(),
                 max_steps=1_000_000):
        """RK4 trajectory and analytic local multiplicative-rate sensitivities.

        Sensitivity j is dC/dm_j at m_j=1, multiplying that reaction's *effective*
        rate (after rate_scales). Initial concentrations are held fixed.
        No clipping, adaptive error estimator, stiffness handling or interpolation.
        """
        step = _number(max_step_s, "max_step_s")
        if not step:
            raise ValueError("max_step_s must be positive")
        times = np.array([_number(t, "time") for t in times_s])
        if not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must start at zero and increase strictly")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        step_counts = []
        for delta in np.diff(times):
            ratio = float(delta) / step
            if not isfinite(ratio) or ratio > max_steps:
                raise ValueError("integration exceeds max_steps budget")
            step_counts.append(max(1, ceil(ratio)))
        if sum(step_counts) > max_steps:
            raise ValueError("integration exceeds max_steps budget")
        selected = tuple(sensitivity_reactions)
        if len(set(selected)) != len(selected) or set(selected) - set(self.reaction_ids):
            raise ValueError("sensitivity reactions must be unique known ids")
        indices = [self.reaction_ids.index(r) for r in selected]
        rates = self._rates({} if rate_scales is None else rate_scales)
        n, p = len(self.ids), len(selected)
        z = np.concatenate((self._state(initial), np.zeros(n*p)))

        def derivative(state):
            x = state[:n]
            if not np.all(np.isfinite(state)) or np.any(x < 0):
                raise ValueError("nonfinite or negative integration state; reduce max_step_s or use another solver")
            with np.errstate(over="raise", invalid="raise"):
                flux = rates * np.prod(x ** self._orders, axis=1)
                dx = self._stoich @ flux
                if not p:
                    return dx
                # Product rule without dividing by x: valid at zero concentration.
                dv = np.zeros((len(self.reactions), n))
                for j, powers in enumerate(self._orders):
                    for i, power in enumerate(powers):
                        if power:
                            reduced = powers.copy()
                            reduced[i] -= 1
                            dv[j, i] = rates[j] * power * np.prod(x ** reduced)
                jacobian = self._stoich @ dv
                source = self._stoich[:, indices] * flux[indices]
                ds = jacobian @ state[n:].reshape(n, p) + source
                return np.concatenate((dx, ds.ravel()))

        values = [z[:n].copy()]
        sensitivities = [z[n:].reshape(n, p).copy()]
        for start, stop, count in zip(times[:-1], times[1:], step_counts):
            h = (stop-start)/count
            for _ in range(count):
                try:
                    k1 = derivative(z)
                    k2 = derivative(z+h*k1/2)
                    k3 = derivative(z+h*k2/2)
                    k4 = derivative(z+h*k3)
                    z = z + h*(k1+2*k2+2*k3+k4)/6
                except FloatingPointError as exc:
                    raise ValueError("integration overflow; reduce step or rescale model") from exc
                if not np.all(np.isfinite(z)) or np.any(z[:n] < 0):
                    raise ValueError("nonfinite or negative integration result; reduce max_step_s or use another solver")
            values.append(z[:n].copy())
            sensitivities.append(z[n:].reshape(n, p).copy())
        return Trajectory(times, self.ids, np.array(values), selected, np.array(sensitivities))


@dataclass(frozen=True)
class Phase:
    """At phase start retain fractions, then add mol/L; evolve for duration_s.

    Omitted retention/addition entries mean one/zero. Rate scales are relative to
    base network rates independently in every phase, never inherited.
    """
    name: str
    duration_s: float
    provenance: tuple[str, ...]
    retain: Mapping[str, float] = field(default_factory=dict)
    add: Mapping[str, float] = field(default_factory=dict)
    rate_scales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        _label(self.name, "phase name")
        object.__setattr__(self, "duration_s", _number(self.duration_s, "duration_s"))
        object.__setattr__(self, "provenance", _sources(self.provenance))
        for name in ("retain", "add", "rate_scales"):
            values = {key: _number(value, name) for key, value in getattr(self, name).items()}
            if name == "retain" and any(v > 1 for v in values.values()):
                raise ValueError("retained fractions cannot exceed one")
            object.__setattr__(self, name, MappingProxyType(values))


@dataclass
class PhaseResult:
    phase: Phase
    start_s: float
    before: dict[str, float]
    after: dict[str, float]
    trajectory: Trajectory


def run_protocol(network, initial, phases, *, max_step_s):
    """Execute explicit ordered phases; preserve both sides of every state jump.

    Each trajectory uses local phase time, with start_s giving absolute offset.
    Conservation applies during dynamics, not across external retain/add events.
    """
    network._state(initial)
    phases = tuple(phases)
    if not phases or not all(isinstance(p, Phase) for p in phases):
        raise ValueError("provide at least one Phase")
    if len({p.name for p in phases}) != len(phases):
        raise ValueError("phase names must be unique")
    # Validate all operations before running any dynamics.
    end_s = 0.0
    for phase in phases:
        end_s = _number(end_s + phase.duration_s, "cumulative protocol time")
        if (set(phase.retain) | set(phase.add)) - set(network.ids):
            raise ValueError("phase state intervention names unknown species")
        network._rates(phase.rate_scales)
    results = []
    state, start = dict(initial), 0.0
    for phase in phases:
        before = dict(state)
        state = {s: state[s]*phase.retain.get(s, 1) + phase.add.get(s, 0) for s in network.ids}
        times = [0, phase.duration_s] if phase.duration_s else [0]
        trajectory = network.simulate(state, times, max_step_s=max_step_s, rate_scales=phase.rate_scales)
        results.append(PhaseResult(phase, start, before, dict(state), trajectory))
        state = trajectory.final
        start += phase.duration_s
    return tuple(results)
