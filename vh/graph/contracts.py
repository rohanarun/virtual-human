"""Explicit scalar contracts for executing a named model behind a graph edge.

This checks declared interfaces, not the truth of a model or its provenance.
Units, reference bases and contexts must match exactly; conversions are models
registered by the caller, never guessed from strings. No engines are registered
automatically and graph deserialization never imports or executes code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from types import MappingProxyType

from .core import Edge


class ContractError(ValueError):
    """An interface is incomplete, incompatible or outside its declared limits."""


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _finite(value: Real, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{label} must be a finite real scalar (not a boolean)")
    try:
        finite = isfinite(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError(f"{label} is outside the supported finite scalar range") from exc
    if not finite:
        raise ContractError(f"{label} must be a finite real scalar (not a boolean)")


def _context(context: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(context, Mapping) or not context:
        raise ContractError("context must be a nonempty mapping of named conditions")
    snapshot = dict(context)
    for key, value in snapshot.items():
        _text(key, "context key")
        _text(value, f"context[{key!r}]")
    return MappingProxyType(snapshot)


def _provenance(provenance: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(provenance, str):
        raise ContractError("provenance must be a sequence of source strings")
    try:
        sources = tuple(provenance)
    except TypeError as exc:
        raise ContractError("provenance must be a sequence of source strings") from exc
    if not sources:
        raise ContractError("provenance must include at least one source")
    for source in sources:
        _text(source, "provenance source")
    return sources


@dataclass(frozen=True)
class Quantity:
    """A finite scalar with explicit semantics and source identifiers.

    ``quantity`` identifies what was measured (e.g. concentration or clearance),
    ``basis`` its reference (e.g. plasma), and ``context`` all conditions the
    caller considers relevant (species, compound, compartment, assay, etc.).
    A missing measurement is absent, not a Quantity with value zero or NaN.
    """

    value: float
    quantity: str
    unit: str
    basis: str
    context: Mapping[str, str]
    provenance: tuple[str, ...]

    def __post_init__(self):
        _finite(self.value, "value")
        for name in ("quantity", "unit", "basis"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "context", _context(self.context))
        object.__setattr__(self, "provenance", _provenance(self.provenance))


@dataclass(frozen=True)
class PortSpec:
    """Required semantics and optional inclusive numerical domain for one port.

    Context is an exact match, including keys. A new species, specimen or assay
    therefore needs a deliberate new contract, even if the units are identical.
    Bounds are in this port's unit and basis. This is not dimensional analysis.
    """

    quantity: str
    unit: str
    basis: str
    context: Mapping[str, str]
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        for name in ("quantity", "unit", "basis"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "context", _context(self.context))
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ContractError("minimum must not exceed maximum")

    def check(self, value: Quantity, label: str) -> None:
        if not isinstance(value, Quantity):
            raise ContractError(f"{label} must be a Quantity")
        _finite(value.value, f"{label}.value")
        for name in ("quantity", "unit", "basis", "context"):
            if getattr(value, name) != getattr(self, name):
                raise ContractError(f"{label}: {name} mismatch")
        if self.minimum is not None and value.value < self.minimum:
            raise ContractError(f"{label}: value below minimum {self.minimum}")
        if self.maximum is not None and value.value > self.maximum:
            raise ContractError(f"{label}: value above maximum {self.maximum}")


def _ports(ports: Mapping[str, PortSpec], label: str) -> Mapping[str, PortSpec]:
    if not isinstance(ports, Mapping):
        raise ContractError(f"{label} must be a mapping of names to PortSpec")
    snapshot = dict(ports)
    for name, spec in snapshot.items():
        _text(name, f"{label} port name")
        if not isinstance(spec, PortSpec):
            raise ContractError(f"{label}[{name!r}] must be a PortSpec")
    return MappingProxyType(snapshot)


@dataclass(frozen=True)
class ModelSpec:
    """A callable and its declared interface; domain/validation are documentation.

    Scientific adequacy of the stated domain, bounds and validation must be
    reviewed separately. Merely providing these strings is not certification.
    """

    key: str
    inputs: Mapping[str, PortSpec]
    outputs: Mapping[str, PortSpec]
    compute: Callable[[Mapping[str, Quantity]], Mapping[str, Quantity]]
    domain: str
    validation: str
    provenance: tuple[str, ...]

    def __post_init__(self):
        for name in ("key", "domain", "validation"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "inputs", _ports(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _ports(self.outputs, "outputs"))
        if not self.outputs:
            raise ContractError("a model must declare at least one output")
        if not callable(self.compute):
            raise ContractError("compute must be callable")
        object.__setattr__(self, "provenance", _provenance(self.provenance))


def _check_ports(values: Mapping[str, Quantity], specs: Mapping[str, PortSpec], label: str) -> dict[str, Quantity]:
    if not isinstance(values, Mapping):
        raise ContractError(f"{label} must be a mapping of named quantities")
    snapshot = dict(values)
    if set(snapshot) != set(specs):
        missing = sorted(set(specs) - set(snapshot))
        extra = sorted(repr(k) for k in set(snapshot) - set(specs))
        raise ContractError(f"{label} ports mismatch: missing={missing}, extra={extra}")
    for name, spec in specs.items():
        spec.check(snapshot[name], f"{label}[{name!r}]")
    return snapshot


class ModelRegistry:
    """Explicit, opt-in dispatch for Edge.model; no graph-wide scheduling.

    Only named inputs are passed to a callable. All inputs are checked first;
    outputs are checked before being returned. A failed output check cannot
    undo callback side effects, so callbacks should be pure functions. This is
    not a security sandbox for untrusted Python.
    """

    def __init__(self):
        self._models: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if not isinstance(spec, ModelSpec):
            raise ContractError("register expects a ModelSpec")
        if spec.key in self._models:
            raise ContractError(f"model {spec.key!r} is already registered")
        self._models[spec.key] = spec

    def execute(self, edge: Edge, inputs: Mapping[str, Quantity]) -> dict[str, Quantity]:
        if not isinstance(edge, Edge) or not isinstance(edge.model, str) or edge.model not in self._models:
            raise ContractError("edge must name an explicitly registered model")
        model = self._models[edge.model]
        checked = _check_ports(inputs, model.inputs, "inputs")
        outputs = _check_ports(model.compute(MappingProxyType(checked)), model.outputs, "outputs")
        # Preserve an audit trail of all inputs and the registered model; these
        # are source identifiers, not an assertion that any source proves output.
        lineage = (*model.provenance, *(source for value in checked.values() for source in value.provenance),
                   f"model:{model.key}")
        return {name: replace(value, provenance=tuple(dict.fromkeys((*value.provenance, *lineage))))
                for name, value in outputs.items()}
