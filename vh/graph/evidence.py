"""Opt-in evidence bindings for scalar models, without scientific certification.

A recorded estimate is not a validated model parameter. Callers must supply the
complete applicable semantics; no source parsing or conversions happen here.
Uncertainty metadata accompanies nominal execution, never becomes a distribution
or an output confidence interval. Missing/conflicting assertions cannot execute.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from types import MappingProxyType
from collections.abc import Mapping
from numbers import Integral, Rational, Real

from .contracts import ContractError, ModelRegistry, PortSpec, Quantity, _text, _finite
from .core import Edge


def _audit_scalar(value):
    """Lossless JSON representation of an accepted scalar, or refuse binding."""
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Rational):
        return {"representation": "rational", "numerator": int(value.numerator),
                "denominator": int(value.denominator)}
    converted = float(value)
    if value != converted:
        raise ContractError("scalar cannot be serialized losslessly for evidence audit")
    return converted


@dataclass(frozen=True)
class Uncertainty:
    """Source-reported interval or explicit lack of quantification.

    CI, IQR and bounds retain their meaning; none implies a sampling distribution.
    Endpoints share the estimate's units and basis. No interval propagation occurs.
    """
    kind: str
    description: str
    low: float | None = None
    high: float | None = None

    def __post_init__(self):
        _text(self.kind, "uncertainty kind")
        _text(self.description, "uncertainty description")
        if (self.low is None) != (self.high is None):
            raise ContractError("uncertainty requires both endpoints or neither")
        if self.low is not None:
            _finite(self.low, "uncertainty low")
            _finite(self.high, "uncertainty high")
            if self.low > self.high:
                raise ContractError("uncertainty endpoints are reversed")


@dataclass(frozen=True)
class EvidenceInput:
    record_id: str
    status: str
    estimate: Quantity | None
    source: str
    locator: str
    limits: str
    uncertainty: Uncertainty

    def __post_init__(self):
        for name in ("record_id", "source", "locator", "limits"):
            _text(getattr(self, name), name)
        if self.status not in ("recorded", "synthetic", "conjecture", "missing", "conflicting"):
            raise ContractError("unknown evidence status")
        if self.estimate is not None and not isinstance(self.estimate, Quantity):
            raise ContractError("estimate must be a Quantity or absent")
        if self.status in ("recorded", "synthetic", "conjecture") and self.estimate is None:
            raise ContractError("numeric evidence status requires an estimate")
        if self.status in ("missing", "conflicting") and self.estimate is not None:
            raise ContractError("missing/conflicting evidence must not select an estimate")
        if not isinstance(self.uncertainty, Uncertainty):
            raise ContractError("uncertainty must be explicit")

    def audit(self) -> dict:
        q = self.estimate
        return {"record_id": self.record_id, "status": self.status, "source": self.source,
                "locator": self.locator, "limits": self.limits,
                "uncertainty": {"kind": self.uncertainty.kind, "description": self.uncertainty.description,
                    "low": None if self.uncertainty.low is None else _audit_scalar(self.uncertainty.low),
                    "high": None if self.uncertainty.high is None else _audit_scalar(self.uncertainty.high)},
                "estimate": None if q is None else {"value": _audit_scalar(q.value), "quantity": q.quantity,
                    "unit": q.unit, "basis": q.basis, "context": dict(q.context),
                    "provenance": list(q.provenance)}}

    def bind(self, spec: PortSpec, *, allow_synthetic: bool = False) -> Quantity:
        if not isinstance(allow_synthetic, bool):
            raise ContractError("allow_synthetic must be boolean")
        if self.status != "recorded" and not (self.status == "synthetic" and allow_synthetic):
            raise ContractError(f"{self.record_id}: evidence status {self.status!r} cannot bind")
        if not isinstance(spec, PortSpec):
            raise ContractError("binding requires a PortSpec")
        spec.check(self.estimate, self.record_id)
        q = self.estimate
        # Structured audit survives ModelRegistry's existing provenance union.
        audit = "evidence:" + json.dumps(self.audit(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return Quantity(q.value, q.quantity, q.unit, q.basis, q.context, (*q.provenance, audit))


def execute_evidence(registry: ModelRegistry, edge: Edge,
                     evidence: Mapping[str, EvidenceInput], specs: Mapping[str, PortSpec], *,
                     allow_synthetic: bool = False) -> dict[str, Quantity]:
    """Validate every binding before dispatch; preserve full evidence in lineage.

    Specs are explicit binding requirements, independently rechecked against the
    registered model by ModelRegistry. This is opt-in and not a security boundary.
    Returned values are nominal scenario outputs, with no inferred uncertainty.
    """
    if not isinstance(evidence, Mapping) or not isinstance(specs, Mapping):
        raise ContractError("evidence and specs must be mappings")
    records, ports = dict(evidence), dict(specs)
    if set(records) != set(ports):
        raise ContractError("evidence ports mismatch")
    if any(not isinstance(v, EvidenceInput) for v in records.values()):
        raise ContractError("all bindings must be EvidenceInput")
    bound = {name: record.bind(ports[name], allow_synthetic=allow_synthetic)
             for name, record in records.items()}
    return registry.execute(edge, MappingProxyType(bound))


@dataclass(frozen=True)
class OutputEnvelope:
    """Deterministic corner-envelope summary for an output port.

    Corner bounds summarize evaluation at the 2^d endpoints of source-reported
    input intervals; they do not define a probability distribution or confidence
    interval, and interior extrema of non-monotonic models are not bounded by
    corner evaluation alone.
    """
    nominal: Quantity
    envelope_low: Quantity
    envelope_high: Quantity
    corner_count: int
    nominal_within_corners: bool
    threshold: float | int | Rational | None = None
    threshold_status: str | None = None


def execute_evidence_envelope(
    registry: ModelRegistry,
    edge: Edge,
    evidence: Mapping[str, EvidenceInput],
    specs: Mapping[str, PortSpec],
    *,
    allow_synthetic: bool = False,
    thresholds: Mapping[str, float | int | Rational] | None = None,
) -> dict[str, OutputEnvelope]:
    """Evaluate a registered scalar model at nominal evidence and all 2^d interval corners.

    Requires every input to carry explicit uncertainty endpoints [low, high] that
    bracket its point estimate (low <= estimate.value <= high) and satisfy its
    PortSpec contract before the model is invoked on any point.
    """
    if not isinstance(evidence, Mapping) or not isinstance(specs, Mapping):
        raise ContractError("evidence and specs must be mappings")
    if thresholds is not None and not isinstance(thresholds, Mapping):
        raise ContractError("thresholds must be a mapping or None")
    records, ports = dict(evidence), dict(specs)
    if set(records) != set(ports):
        raise ContractError("evidence ports mismatch")
    if any(not isinstance(v, EvidenceInput) for v in records.values()):
        raise ContractError("all bindings must be EvidenceInput")

    threshold_map = {} if thresholds is None else dict(thresholds)
    for out_name, tau in threshold_map.items():
        _text(out_name, "threshold port")
        _finite(tau, f"{out_name} threshold")
        _audit_scalar(tau)

    ordered_names = sorted(records)
    bound_nominal: dict[str, Quantity] = {}
    endpoint_choices: list[tuple[Quantity, ...]] = []

    for name in ordered_names:
        rec = records[name]
        spec = ports[name]
        nom_q = rec.bind(spec, allow_synthetic=allow_synthetic)
        unc = rec.uncertainty
        if unc.low is None or unc.high is None:
            raise ContractError(f"{rec.record_id}: envelope execution requires explicit uncertainty endpoints")
        _audit_scalar(unc.low)
        _audit_scalar(unc.high)
        if not (unc.low <= nom_q.value <= unc.high):
            raise ContractError(
                f"{rec.record_id}: point estimate {nom_q.value} lies outside uncertainty interval [{unc.low}, {unc.high}]"
            )
        low_q = Quantity(unc.low, nom_q.quantity, nom_q.unit, nom_q.basis, nom_q.context, nom_q.provenance)
        high_q = Quantity(unc.high, nom_q.quantity, nom_q.unit, nom_q.basis, nom_q.context, nom_q.provenance)
        spec.check(low_q, f"{rec.record_id}.low")
        spec.check(high_q, f"{rec.record_id}.high")
        bound_nominal[name] = nom_q
        endpoint_choices.append((low_q,) if unc.low == unc.high else (low_q, high_q))

    nominal_out = registry.execute(edge, MappingProxyType(bound_nominal))
    unknown_thresholds = set(threshold_map) - set(nominal_out)
    if unknown_thresholds:
        raise ContractError(f"threshold specified for unknown output port: {sorted(unknown_thresholds)[0]}")

    corner_outputs: list[dict[str, Quantity]] = []
    for combo in product(*endpoint_choices):
        corner_in = {name: q for name, q in zip(ordered_names, combo)}
        corner_outputs.append(registry.execute(edge, MappingProxyType(corner_in)))

    envelopes: dict[str, OutputEnvelope] = {}
    for out_name, nom_q in nominal_out.items():
        series = [c[out_name] for c in corner_outputs]
        low_q = min(series, key=lambda q: q.value)
        high_q = max(series, key=lambda q: q.value)
        within = bool(low_q.value <= nom_q.value <= high_q.value)
        tau = threshold_map.get(out_name)
        status = None
        if tau is not None:
            if low_q.value > tau:
                status = "above_across_envelope"
            elif high_q.value < tau:
                status = "below_across_envelope"
            else:
                status = "straddles_threshold"
        envelopes[out_name] = OutputEnvelope(
            nominal=nom_q,
            envelope_low=low_q,
            envelope_high=high_q,
            corner_count=len(corner_outputs),
            nominal_within_corners=within,
            threshold=tau,
            threshold_status=status,
        )
    return envelopes
