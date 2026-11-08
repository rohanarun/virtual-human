"""SBML import: analytic kinetics, exact stoichiometry, refusals, and a referee.

Synthetic networks below are artificial fixtures with closed-form solutions.
The vendored BioModels (data/sbml, CC0) are checked against exact algebra and,
when installed, against libRoadRunner -- an independent SBML implementation
(libSBML + CVODE) used here only as a referee.
"""

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vh.graph import Graph
from vh.graph import stoichiometry as S
from vh.graph.sbml import SBMLError, UnsupportedSBML, read_sbml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sbml"
M = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def l3(body: str, version: int = 2, extra: str = "") -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<sbml xmlns="http://www.sbml.org/sbml/level3/'
            f'version{version}/core" level="3" version="{version}"{extra}><model id="m">{body}</model></sbml>'
            ).encode()


def l2(body: str) -> bytes:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<sbml xmlns="http://www.sbml.org/sbml/level2/version4" '
            f'level="2" version="4"><model id="m">{body}</model></sbml>').encode()


def species(sid, comp="c", conc=None, amount=None, hosu="false", boundary="false", constant="false"):
    init = f'initialConcentration="{conc}"' if conc is not None else f'initialAmount="{amount}"'
    return (f'<species id="{sid}" compartment="{comp}" {init} hasOnlySubstanceUnits="{hosu}" '
            f'boundaryCondition="{boundary}" constant="{constant}"/>')


def reaction(rid, reactants, products, math, local="", reversible="false", modifiers=()):
    def refs(tag, items):
        if not items:
            return ""
        inner = "".join(f'<speciesReference species="{s}" stoichiometry="{c}" constant="true"/>' for s, c in items)
        return f"<{tag}>{inner}</{tag}>"
    mods = ("<listOfModifiers>" + "".join(f'<modifierSpeciesReference species="{m}"/>' for m in modifiers)
            + "</listOfModifiers>") if modifiers else ""
    lp = f"<listOfLocalParameters>{local}</listOfLocalParameters>" if local else ""
    return (f'<reaction id="{rid}" reversible="{reversible}">{refs("listOfReactants", reactants)}'
            f'{refs("listOfProducts", products)}{mods}<kineticLaw><math {M}>{math}</math>{lp}</kineticLaw></reaction>')


def apply(op, *args):
    return f"<apply><{op}/>{''.join(args)}</apply>"


def ci(x):
    return f"<ci>{x}</ci>"


def cn(x):
    return f"<cn>{x}</cn>"


COMP = '<listOfCompartments><compartment id="c" size="{size}" constant="true"/></listOfCompartments>'


# ---------------------------------------------------------------------------
# kinetics against closed-form solutions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hosu", ["false", "true"])
def test_first_order_decay_respects_concentration_versus_amount(hosu):
    # V = 2.5, A0 = 10 (amount). Kinetic laws are substance/time: with
    # hasOnlySubstanceUnits=false the symbol A is a concentration, so the law
    # k*c*A and the amount-based law k*A describe the same amount decay.
    size, k = 2.5, 0.3
    law = apply("times", ci("k"), ci("c"), ci("A")) if hosu == "false" else apply("times", ci("k"), ci("A"))
    doc = l3(COMP.format(size=size)
             + "<listOfSpecies>" + species("A", amount=10, hosu=hosu) + species("B", amount=0, hosu=hosu)
             + "</listOfSpecies><listOfReactions>"
             + reaction("decay", [("A", 1)], [("B", 1)], law, f'<localParameter id="k" value="{k}"/>')
             + "</listOfReactions>")
    m = read_sbml(doc)
    t = np.linspace(0, 20, 41)
    out = m.simulate(t)
    exact = 10 * np.exp(-k * t)
    np.testing.assert_allclose(out["amount"][:, 0], exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(out["amount"].sum(axis=1), 10, rtol=1e-12)
    expected_value = exact / size if hosu == "false" else exact
    np.testing.assert_allclose(out["value"][:, 0], expected_value, rtol=1e-8, atol=1e-10)
    assert m.conservation_laws() == [{"A": 1, "B": 1}]


def test_reversible_mass_action_relaxation_steady_state_and_eigenvalue():
    kf, kr, a0 = 2.0, 0.5, 3.0
    law = apply("minus", apply("times", ci("kf"), ci("A")), apply("times", ci("kr"), ci("B")))
    doc = l3(COMP.format(size=1) + '<listOfParameters><parameter id="kf" value="2" constant="true"/>'
             '<parameter id="kr" value="0.5" constant="true"/></listOfParameters><listOfSpecies>'
             + species("A", conc=a0) + species("B", conc=0) + "</listOfSpecies><listOfReactions>"
             + reaction("r", [("A", 1)], [("B", 1)], law, reversible="true") + "</listOfReactions>")
    m = read_sbml(doc)
    a_eq = a0 * kr / (kf + kr)
    t = np.linspace(0, 5, 26)
    out = m.simulate(t)
    np.testing.assert_allclose(out["amount"][:, 0], a_eq + (a0 - a_eq) * np.exp(-(kf + kr) * t), rtol=1e-8)
    ss = m.steady_state()
    assert ss["success"]
    np.testing.assert_allclose(ss["amount"], [a_eq, a0 - a_eq], rtol=1e-12)
    np.testing.assert_allclose(m.reduced_eigenvalues(ss["amount"]), [-(kf + kr)], rtol=1e-6)


def test_functions_piecewise_time_rules_initial_assignment_and_local_shadowing():
    body = (
        f'<listOfFunctionDefinitions><functionDefinition id="sat"><math {M}><lambda>'
        f'<bvar>{ci("x")}</bvar><bvar>{ci("K")}</bvar>{apply("divide", ci("x"), apply("plus", ci("K"), ci("x")))}'
        '</lambda></math></functionDefinition></listOfFunctionDefinitions>'
        + COMP.format(size=2)
        + '<listOfParameters><parameter id="k" value="100" constant="true"/>'
          '<parameter id="g" constant="false"/><parameter id="base" value="1.5" constant="true"/></listOfParameters>'
        + "<listOfSpecies>" + species("A", conc=1) + species("B", conc=0) + "</listOfSpecies>"
        + f'<listOfInitialAssignments><initialAssignment symbol="A"><math {M}>'
          f'{apply("times", cn(3), ci("base"))}</math></initialAssignment></listOfInitialAssignments>'
        + f'<listOfRules><assignmentRule variable="g"><math {M}>'
          f'<piecewise><piece>{cn(2)}{apply("lt", "<csymbol encoding=\'text\' definitionURL=\'http://www.sbml.org/sbml/symbols/time\'>t</csymbol>", cn(5))}</piece>'
          f'<otherwise>{cn(0)}</otherwise></piecewise></math></assignmentRule></listOfRules>'
        + "<listOfReactions>"
        + reaction("r", [("A", 1)], [("B", 1)],
                   apply("times", ci("c"), ci("k"), ci("g"), f"<apply>{ci('sat')}{ci('A')}{cn(0.5)}</apply>"),
                   '<localParameter id="k" value="0.25"/>')
        + "</listOfReactions>")
    m = read_sbml(l3(body))
    x0 = m.initial_amounts()
    np.testing.assert_allclose(x0, [3 * 1.5 * 2, 0])          # initial assignment is a concentration
    a = 4.5
    expected = 2 * 0.25 * 2 * a / (0.5 + a)                    # local k shadows the global k = 100
    assert m.rates(x0, t=1.0)[0] == pytest.approx(expected, rel=1e-15)
    assert m.rates(x0, t=6.0)[0] == 0.0                        # piecewise on csymbol time


# ---------------------------------------------------------------------------
# refusals: unsupported constructs are named, never silently dropped
# ---------------------------------------------------------------------------
BASE = COMP.format(size=1) + "<listOfSpecies>" + species("A", conc=1) + species("B", conc=0) + "</listOfSpecies>"
DECAY = reaction("r", [("A", 1)], [("B", 1)], apply("times", cn(0.1), ci("A")))


@pytest.mark.parametrize("doc,message", [
    (l3(BASE + "<listOfReactions>" + DECAY + "</listOfReactions><listOfEvents><event useValuesFromTriggerTime='true'/></listOfEvents>"), "listOfEvents"),
    (l3(BASE + f'<listOfRules><rateRule variable="A"><math {M}>{cn(1)}</math></rateRule></listOfRules>'), "rateRule"),
    (l3(BASE + f'<listOfRules><algebraicRule><math {M}>{ci("A")}</math></algebraicRule></listOfRules>'), "algebraicRule"),
    (l3(BASE + "<listOfReactions>" + reaction("r", [("A", 1)], [("B", 1)],
        "<apply><csymbol definitionURL='http://www.sbml.org/sbml/symbols/delay'>d</csymbol>"
        + ci("A") + cn(1) + "</apply>") + "</listOfReactions>"), "delay"),
    (l3(BASE + "<listOfReactions>" + DECAY.replace('reversible="false"', 'reversible="false" fast="true"')
        + "</listOfReactions>", version=1), "fast reaction"),
    (l2('<listOfCompartments><compartment id="c" size="1"/></listOfCompartments><listOfSpecies>'
        '<species id="A" compartment="c" initialAmount="1"/><species id="B" compartment="c" initialAmount="0"/>'
        '</listOfSpecies><listOfReactions><reaction id="r"><listOfReactants><speciesReference species="A">'
        f'<stoichiometryMath><math {M}>{cn(2)}</math></stoichiometryMath></speciesReference></listOfReactants>'
        f'<kineticLaw><math {M}>{ci("A")}</math></kineticLaw></reaction></listOfReactions>'), "stoichiometryMath"),
    (l3(BASE, extra=' xmlns:fbc="http://www.sbml.org/sbml/level3/version1/fbc/version2" fbc:required="true"'),
     "required SBML package"),
    (l3(BASE.replace('<species id="A"', '<species conversionFactor="f" id="A"')), "conversionFactor"),
    (l3(BASE + "<listOfReactions>" + reaction("r", [("A", 1)], [("B", 1)], apply("factorial", ci("A")))
        + "</listOfReactions>"), "factorial"),
    (l3(BASE + "<listOfReactions>" + reaction("r", [("A", 1)], [("B", 1)], ci("r")) + "</listOfReactions>"),
     "reaction id"),
    (l3(BASE.replace('constant="true"/></listOfCompartments>', 'constant="false"/></listOfCompartments>')),
     "non-constant compartment"),
    (l3(BASE + "<listOfConstraints><constraint/></listOfConstraints>"), "listOfConstraints"),
])
def test_unsupported_constructs_are_refused_by_name(doc, message):
    with pytest.raises(UnsupportedSBML, match=message):
        read_sbml(doc)


@pytest.mark.parametrize("prefix", [
    b'<?xml version="1.0"?><!DOCTYPE sbml [<!ENTITY a "aaaa">]>',
    b'<?xml version="1.0"?><!DOCTYPE sbml SYSTEM "file:///etc/passwd">',
])
def test_doctype_and_entity_declarations_are_refused(prefix):
    with pytest.raises(SBMLError, match="DOCTYPE"):
        read_sbml(prefix + l3(BASE).split(b"\n", 1)[1])


UTF16_ENTITY = ('<?xml version="1.0" encoding="UTF-16"?>\n<!DOCTYPE sbml [<!ENTITY x "EXPANDED_VALUE">]>\n'
                '<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="m">'
                '<listOfCompartments><compartment id="c" name="&x;" size="1"/></listOfCompartments></model></sbml>')


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_entity_refusal_does_not_depend_on_the_byte_encoding(encoding):
    # Regression: an ASCII byte scan missed these, and the entity was expanded.
    with pytest.raises(SBMLError, match="DOCTYPE"):
        read_sbml(UTF16_ENTITY.encode(encoding))


def test_utf16_documents_without_a_doctype_still_load():
    doc = l3(BASE + "<listOfReactions>" + DECAY + "</listOfReactions>").decode()
    m = read_sbml(doc.replace('encoding="UTF-8"', 'encoding="UTF-16"').encode("utf-16"))
    assert m.dynamic_species == ["A", "B"]


@pytest.mark.parametrize("data", [b"", b"<sbml", "<?xml version='1.0' encoding='UTF-32'?><sbml/>".encode("utf-32")])
def test_unreadable_xml_is_an_sbml_error(data):
    with pytest.raises(SBMLError):
        read_sbml(data)


@pytest.mark.parametrize("doc,message", [
    (l3(COMP.format(size=1) + "<listOfSpecies>" + species("A", conc=1, constant="true") + species("B", conc=0)
        + "</listOfSpecies><listOfReactions>" + DECAY + "</listOfReactions>"), "constant"),
    (l3(BASE + '<listOfParameters><parameter id="p" constant="false"/><parameter id="q" constant="false"/>'
        f'</listOfParameters><listOfRules><assignmentRule variable="p"><math {M}>{ci("q")}</math></assignmentRule>'
        f'<assignmentRule variable="q"><math {M}>{ci("p")}</math></assignmentRule></listOfRules>'), "algebraic loop"),
    (l3(BASE + "<listOfReactions>" + reaction("r", [("A", 1)], [("B", 1)], ci("nope")) + "</listOfReactions>"),
     "unknown symbols"),
    (l3(COMP.format(size=1) + '<listOfSpecies><species id="A" compartment="c" hasOnlySubstanceUnits="false" '
        'boundaryCondition="false" constant="false"/></listOfSpecies>'), "no initial value"),
    (l3(COMP.format(size=1) + '<listOfSpecies><species id="A" compartment="c" initialAmount="1"/></listOfSpecies>'),
     "needs attribute"),
    (l3(BASE + "<listOfReactions>" + reaction("r", [("A", 0)], [("B", 1)], ci("A")) + "</listOfReactions>"),
     "positive"),
])
def test_semantic_errors_are_reported(doc, message):
    with pytest.raises(SBMLError, match=message):
        read_sbml(doc)


# ---------------------------------------------------------------------------
# exact stoichiometric algebra
# ---------------------------------------------------------------------------
def test_exact_rref_nullspace_and_conservation_laws():
    # A + E <-> AE -> B + E : conserved E + AE and A + AE + B.
    n = [[-1, 1, 0], [-1, 1, 1], [1, -1, -1], [0, 0, 1]]     # rows A, E, AE, B
    laws = S.conservation_laws(n)
    assert laws == [[1, 0, 1, 1], [0, 1, 1, 0]]
    assert S.conservation_residual(laws, n) == 0
    assert S.rank(n) + len(laws) == 4
    for v in S.nullspace(n):
        assert all(sum(Fraction(row[j]) * v[j] for j in range(3)) == 0 for row in n)
    assert S.integer_row([Fraction(-1, 2), Fraction(1, 3)]) == [3, -2]
    with pytest.raises(TypeError):
        S.as_fraction(0.1)


def _check_certificates(n, res):
    """Independent exact check of the returned certificates."""
    internal = res["internal_reactions"]
    sub = [[Fraction(row[j]) for j in internal] for row in n]
    mass = res["mass"]
    assert all(m >= 0 for m in mass) and all(mass[i] > 0 for i in res["conserved"])
    assert all(sum(mass[i] * sub[i][k] for i in range(len(n))) == 0 for k in range(len(internal)))
    if res["unconserved"]:
        v = res["leak_combination"]
        produced = [sum(sub[i][k] * v[k] for k in range(len(internal))) for i in range(len(n))]
        assert all(p >= 0 for p in produced) and all(produced[i] > 0 for i in res["unconserved"])


@pytest.mark.parametrize("n,internal,unconserved", [
    ([[-1, 0], [-1, 0], [1, -1], [0, 1]], None, []),         # A + B -> C -> D
    # A -> A + B: the net column hides the catalytic A, so the caller (as
    # SBMLModel does from reactant/product lists) marks the reaction internal.
    ([[0], [1]], [True], [1]),
    ([[0], [1]], None, []),                                   # same column read as pure synthesis
    ([[-1], [-1], [1]], None, []),                            # A + B -> C
    ([[-1, 0], [0, 1]], None, []),                            # A -> (out), (in) -> B: exchanges only
    ([[-1, 1], [1, -2]], None, [0, 1]),                       # A -> B, 2B -> A: mass cannot balance
    ([[-2, 1], [1, 0], [0, -1]], None, []),                   # 2A -> B, C -> A (fractional masses)
])
def test_stoichiometric_consistency_with_exact_certificates(n, internal, unconserved):
    res = S.stoichiometric_consistency(n, internal)
    assert res["unconserved"] == unconserved
    assert res["consistent"] == (not unconserved)
    assert res["certified"]
    _check_certificates(n, res)


# ---------------------------------------------------------------------------
# vendored BioModels: provenance, exact structure and graph invariants
# ---------------------------------------------------------------------------
SOURCES = json.loads((DATA / "sources.json").read_text())
EXPECTED_LAWS = {
    # Kinase totals of the three-tier cascade (Kholodenko 2000).
    "BIOMD0000000010": [{"MKKK": 1, "MKKK_P": 1}, {"MKK": 1, "MKK_P": 1, "MKK_PP": 1},
                        {"MAPK": 1, "MAPK_P": 1, "MAPK_PP": 1}],
    # Two enzymes, three kinases and two phosphatases, each free plus bound (Huang & Ferrell 1996).
    "BIOMD0000000009": [
        {"E1": 1, "E1_KKK": 1}, {"E2": 1, "E2_P_KKK": 1},
        {"KKK": 1, "P_KKK": 1, "E1_KKK": 1, "E2_P_KKK": 1, "P_KKK_KK": 1, "P_KKK_P_KK": 1},
        {"KK": 1, "P_KK": 1, "PP_KK": 1, "P_KKK_KK": 1, "P_KKK_P_KK": 1, "PP_KK_K": 1, "PP_KK_P_K": 1,
         "KKPase_PP_KK": 1, "KKPase_P_KK": 1},
        {"K": 1, "P_K": 1, "PP_K": 1, "PP_KK_K": 1, "PP_KK_P_K": 1, "KPase_PP_K": 1, "KPase_P_K": 1},
        {"KPase": 1, "KPase_PP_K": 1, "KPase_P_K": 1}, {"KKPase": 1, "KKPase_PP_KK": 1, "KKPase_P_KK": 1}],
    # Synthesis and degradation of every species: nothing is conserved (Elowitz & Leibler 2000).
    "BIOMD0000000012": [],
}


def load(mid):
    return read_sbml(DATA / f"{mid}.xml")


def test_vendored_models_match_recorded_provenance():
    assert {m["id"] for m in SOURCES["models"]} == set(EXPECTED_LAWS)
    for entry in SOURCES["models"]:
        data = (DATA / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"]
        assert entry["license"].startswith("CC0") and b"CC0" in data


@pytest.mark.parametrize("mid", sorted(EXPECTED_LAWS))
def test_biomodels_exact_conservation_consistency_and_graph(mid):
    m = load(mid)
    assert m.conservation_laws() == EXPECTED_LAWS[mid]
    n = m.stoichiometry_matrix()
    laws = S.conservation_laws(n)
    assert S.conservation_residual(laws, n) == 0 and S.rank(n) + len(laws) == len(m.dynamic_species)
    res = m.stoichiometric_consistency()
    assert res["consistent"] and res["certified"]
    g, schema = m.to_graph()
    assert g.validate(schema) == []
    restored = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert restored.validate(schema) == []


def test_kholodenko_totals_are_conserved_along_the_trajectory():
    m = load("BIOMD0000000010")
    out = m.simulate(np.linspace(0, 3000, 61))
    idx = {s: i for i, s in enumerate(out["species"])}
    for law, total in zip(EXPECTED_LAWS["BIOMD0000000010"], (100, 300, 300)):
        series = sum(c * out["amount"][:, idx[s]] for s, c in law.items())
        np.testing.assert_allclose(series, total, rtol=1e-9)


def test_graph_invariants_catch_tampered_structure():
    g, schema = load("BIOMD0000000009").to_graph()
    member = next(e for e in g.edges_of_type("member") if e.src == "E1_KKK")
    member.attrs["coefficient"] = "2"
    assert any("[conservation] moiety_1: not conserved" in p for p in g.validate(schema))

    g, schema = load("BIOMD0000000009").to_graph()
    g = g.subgraph([n for n in g.nodes if n != "moiety_7"])
    assert any("6 moieties recorded" in p for p in g.validate(schema))

    g, schema = load("BIOMD0000000010").to_graph()
    next(iter(g.edges_of_type("product"))).attrs["stoichiometry"] = "0"
    assert any("[stoichiometry]" in p for p in g.validate(schema))

    # A product edge that creates MAPK from a conversion of MKKK breaks both
    # the recorded moieties and positive-mass consistency.
    g, schema = load("BIOMD0000000010").to_graph()
    g.add_edge("J0", "MAPK", "product", stoichiometry="1")
    problems = g.validate(schema)
    assert any("[conservation]" in p for p in problems)
    assert any("[stoichiometric-consistency]" in p and "exact certificate" in p for p in problems)


# ---------------------------------------------------------------------------
# independent referee: libRoadRunner
# ---------------------------------------------------------------------------
TIMES = {"BIOMD0000000010": 4000.0, "BIOMD0000000009": 200.0, "BIOMD0000000012": 1000.0}
# Presimulation before a steady-state solve: long enough to settle (stable
# model) or to sample a whole orbit (oscillators). Huang-Ferrell amounts are
# ~1e-13, so the referee's absolute steady-state threshold needs a start near
# the fixed point.
SETTLE = {"BIOMD0000000010": 4000.0, "BIOMD0000000009": 2000.0, "BIOMD0000000012": 1000.0}


def _referee(mid):
    roadrunner = pytest.importorskip("roadrunner")
    rr = roadrunner.RoadRunner(str(DATA / f"{mid}.xml"))
    rr.integrator.relative_tolerance = 1e-12
    rr.integrator.absolute_tolerance = 1e-16
    rr.integrator.maximum_num_steps = 100000
    return rr


@pytest.mark.parametrize("mid", sorted(TIMES))
def test_trajectories_match_libroadrunner(mid):
    rr = _referee(mid)
    m = load(mid)
    ids = m.dynamic_species
    ref = np.array(rr.simulate(0, TIMES[mid], 101, selections=["time"] + ids))
    ours = m.simulate(ref[:, 0])
    scale = np.abs(ref[:, 1:]).max(axis=0)
    assert np.max(np.abs(ours["amount"] - ref[:, 1:]) / scale) < 1e-8


@pytest.mark.parametrize("mid", sorted(TIMES))
def test_conservation_space_matches_libroadrunner(mid):
    rr = _referee(mid)
    rr.conservedMoietyAnalysis = True
    m = load(mid)
    ids = m.dynamic_species
    cm = rr.getConservationMatrix()
    ref = np.array(cm)[:, [list(cm.colnames).index(s) for s in ids]] if np.array(cm).size else np.zeros((0, len(ids)))
    ref = ref[np.abs(ref).sum(axis=1) > 0]                    # rows on rule-set species vanish here
    ours = np.array([[law.get(s, 0) for s in ids] for law in m.conservation_laws()], float).reshape(-1, len(ids))
    rank = np.linalg.matrix_rank
    assert rank(ours) == rank(ref) == (rank(np.vstack([ours, ref])) if len(ours) else 0)


@pytest.mark.parametrize("mid,stable", [("BIOMD0000000009", True), ("BIOMD0000000010", False),
                                        ("BIOMD0000000012", False)])
def test_steady_state_and_stability_match_libroadrunner(mid, stable):
    from scipy.optimize import linear_sum_assignment

    rr = _referee(mid)
    rr.conservedMoietyAnalysis = True
    m = load(mid)
    ids = m.dynamic_species
    tail = m.simulate(np.linspace(0, SETTLE[mid], 401))["amount"][200:]
    ss = m.steady_state(tail.mean(axis=0))                   # an orbit's mean starts Newton near the fixed point
    assert ss["success"] and ss["max_abs_derivative"] < 1e-9
    rr.simulate(0, SETTLE[mid], 3)
    rr.steadyState()
    ref = np.array([rr[s] for s in ids])
    assert np.max(np.abs(ss["amount"] - ref)) / np.max(np.abs(ref)) < 1e-7
    ours_ev = m.reduced_eigenvalues(ss["amount"])
    ref_ev = np.array(rr.getReducedEigenValues())
    cost = np.abs(ours_ev[:, None] - ref_ev[None, :])
    rows, cols = linear_sum_assignment(cost)
    assert len(ours_ev) == len(ref_ev)
    assert np.max(cost[rows, cols]) / np.max(np.abs(ref_ev)) < 1e-6
    assert (ours_ev.real.max() < 0) == stable
