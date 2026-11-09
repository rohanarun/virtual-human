"""Exact stoichiometric analysis: conservation laws and stoichiometric consistency.

Stoichiometric coefficients are exact rationals (``fractions.Fraction``), so the
linear algebra here has no rounding: a conservation law ``l`` is accepted only if
``l . N`` is exactly zero, and the number of laws is exactly ``n - rank(N)``.

Stoichiometric consistency (Gevorgyan, Poolman & Fell, Bioinformatics 24:2245,
2008) asks whether every species can be given a strictly positive mass such that
every internal reaction conserves mass. It is a linear program; the LP is solved
in floating point (SciPy/HiGHS), then each answer is turned into an exact
rational certificate and checked exactly:

  - a conserved species gets a nonnegative mass vector ``m`` with
    ``m^T N_internal = 0`` exactly and ``m_i > 0``;
  - an unconserved species gets a reaction combination ``v`` with
    ``N_internal v >= 0`` exactly and ``(N_internal v)_i > 0``. For any
    nonnegative ``m`` with ``m^T N_internal = 0``, ``0 = m^T N v >= m_i (N v)_i``,
    so ``m_i = 0``: no positive mass exists (Farkas' lemma).

If rounding prevents an exact certificate, the result says so instead of
reporting the floating-point answer as proven.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm
from numbers import Rational

Matrix = list[list[Fraction]]


def as_fraction(value) -> Fraction:
    """Exact rational from an int, Fraction or decimal string; floats are rejected
    because their binary value is rarely the coefficient the author meant."""
    if isinstance(value, bool):
        raise TypeError("a boolean is not a stoichiometric coefficient")
    if isinstance(value, (Rational, str)):
        return Fraction(value)
    raise TypeError(f"exact rational required, got {type(value).__name__}")


def _matrix(rows) -> Matrix:
    out = [[as_fraction(x) for x in row] for row in rows]
    if out and any(len(r) != len(out[0]) for r in out):
        raise ValueError("matrix rows must have equal length")
    return out


def transpose(rows: Matrix, ncols: int | None = None) -> Matrix:
    if not rows:
        return [[] for _ in range(ncols or 0)]
    return [list(col) for col in zip(*rows)]


def rref(rows) -> tuple[Matrix, list[int]]:
    """Reduced row echelon form over the rationals, and its pivot columns."""
    a = [list(r) for r in _matrix(rows)]
    if not a:
        return [], []
    nrows, ncols = len(a), len(a[0])
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, nrows) if a[i][c] != 0), None)
        if p is None:
            continue
        a[r], a[p] = a[p], a[r]
        inv = 1 / a[r][c]
        a[r] = [x * inv for x in a[r]]
        for i in range(nrows):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == nrows:
            break
    return a[:r], pivots


def rank(rows) -> int:
    return len(rref(rows)[1])


def nullspace(rows, ncols: int | None = None) -> Matrix:
    """Basis of {v : A v = 0} (one vector per free column of the RREF)."""
    a = _matrix(rows)
    n = len(a[0]) if a else (ncols or 0)
    r, pivots = rref(a)
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for row, p in zip(r, pivots):
            v[p] = -row[f]
        basis.append(v)
    return basis


def integer_row(row) -> list[int]:
    """Scale a rational row to coprime integers with a positive first nonzero."""
    row = [as_fraction(x) for x in row]
    den = lcm(*(x.denominator for x in row)) if row else 1
    ints = [int(x * den) for x in row]
    g = gcd(*ints) if any(ints) else 1
    ints = [x // g for x in ints]
    first = next((x for x in ints if x), 0)
    return [-x for x in ints] if first < 0 else ints


def conservation_laws(stoichiometry) -> list[list[int]]:
    """Canonical integer basis of the left nullspace of N (species x reactions).

    Each row ``l`` satisfies ``l . N = 0`` exactly, so ``sum_i l_i x_i`` is
    constant along every trajectory of ``dx/dt = N v(x)``. The basis is the RREF
    of the left nullspace (unique for the space), scaled to coprime integers, so
    two implementations can be compared row for row.
    """
    n = _matrix(stoichiometry)
    nspecies = len(n)
    nreactions = len(n[0]) if n else 0
    if nreactions == 0:
        return [integer_row([int(i == j) for j in range(nspecies)]) for i in range(nspecies)]
    left = nullspace(transpose(n), nspecies)
    if not left:
        return []
    canonical, _ = rref(left)
    return [integer_row(r) for r in canonical]


def conservation_residual(laws, stoichiometry) -> Fraction:
    """max |l . N_j| over laws and reactions, computed exactly (0 means proven)."""
    n = _matrix(stoichiometry)
    worst = Fraction(0)
    for law in laws:
        law = [as_fraction(x) for x in law]
        for j in range(len(n[0]) if n else 0):
            worst = max(worst, abs(sum(law[i] * n[i][j] for i in range(len(n)))))
    return worst


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a] if b and b[0] else [[] for _ in a]


def _certify(b: Matrix, positive: list[int], max_rounds: int = 64):
    """Exact y with w = B y >= 0 and w_i > 0 for every i in ``positive``.

    Solves an LP in floating point (maximise the number of positive entries
    requested), rationalises the answer and checks it exactly. Entries that
    the LP leaves numerically at zero are then constrained to be exactly zero
    by restricting y to the exact nullspace of those rows, and the LP is
    solved again; each round fixes at least one more row, so the loop ends.
    Returns (y, w) as Fractions, or None when no exact certificate is found.
    """
    from scipy.optimize import linprog

    nrows = len(b)
    ncols = len(b[0]) if b else 0
    zero: set[int] = set()
    for _ in range(max_rounds):
        if zero:
            k = nullspace([b[i] for i in sorted(zero)], ncols)
            if not k:
                return None
            basis = transpose(k)                     # ncols x dim
        else:
            basis = [[Fraction(int(i == j)) for j in range(ncols)] for i in range(ncols)]
        m = _matmul(b, basis)                       # nrows x dim
        dim = len(basis[0]) if basis else 0
        free_rows = [i for i in range(nrows) if i not in zero]
        want = [i for i in positive if i not in zero]
        if len(want) < len(positive):
            return None
        mf = [[float(x) for x in m[i]] for i in range(nrows)]
        # variables [y (dim, free), z (len(want), in [0,1])]: M y >= 0 on free rows; z_k <= (M y)_want_k
        a_ub = [[-x for x in mf[i]] + [0.0] * len(want) for i in free_rows]
        a_ub += [[-x for x in mf[i]] + [1.0 if c == k else 0.0 for c in range(len(want))]
                 for k, i in enumerate(want)]
        lp = linprog(c=[0.0] * dim + [-1.0] * len(want), A_ub=a_ub or None,
                     b_ub=[0.0] * len(a_ub) or None,
                     bounds=[(None, None)] * dim + [(0, 1)] * len(want), method="highs")
        if lp.status != 0:
            return None
        y = lp.x[:dim]
        wf = [sum(mf[i][c] * y[c] for c in range(dim)) for i in range(nrows)]
        scale = max((abs(x) for x in wf), default=0.0) or 1.0
        yq = [Fraction(float(v) / scale).limit_denominator(10**9) for v in y]
        w = [sum(m[i][c] * yq[c] for c in range(dim)) for i in range(nrows)]
        if all(x >= 0 for x in w) and all(w[i] > 0 for i in positive):
            return [sum(basis[j][c] * yq[c] for c in range(dim)) for j in range(ncols)], w
        tiny = {i for i in free_rows if i not in positive and abs(wf[i]) / scale < 1e-9}
        if not tiny - zero:
            return None
        zero |= tiny
    return None


def stoichiometric_consistency(stoichiometry, internal: list[bool] | None = None) -> dict:
    """Which species can carry a strictly positive conserved mass?

    Rows of ``stoichiometry`` should be every species that takes part in a
    reaction, including boundary or constant species: holding a species fixed
    changes its dynamics, not its mass. ``internal[j]`` marks reactions that
    should conserve mass; by default a reaction is internal when it has both a
    consumed and a produced species (pure synthesis/degradation/exchange
    reactions are open by construction).

    Returns ``consistent`` (every species conservable), the conserved and
    unconserved species indices, the exact certificates described in the
    module docstring, and ``certified`` (True only if every classification has
    an exactly verified certificate).
    """
    from scipy.optimize import linprog

    n = _matrix(stoichiometry)
    ns = len(n)
    nr = len(n[0]) if n else 0
    if internal is None:
        internal = [any(n[i][j] < 0 for i in range(ns)) and any(n[i][j] > 0 for i in range(ns))
                    for j in range(nr)]
    cols = [j for j in range(nr) if internal[j]]
    sub = [[n[i][j] for j in cols] for i in range(ns)]
    result = {"internal_reactions": cols, "n_species": ns}
    if ns == 0 or not cols:
        return {**result, "consistent": True, "conserved": list(range(ns)), "unconserved": [],
                "mass": [Fraction(1)] * ns, "leak_combination": [], "certified": True}

    # Classification LP: maximise the number of species with positive mass.
    # variables [m (ns, >= 0), z (ns, in [0, 1])]; m^T N_int = 0; z <= m.
    a_eq = [[float(sub[i][k]) for i in range(ns)] + [0.0] * ns for k in range(len(cols))]
    a_ub = [[-1.0 if c == i else 0.0 for c in range(ns)] + [1.0 if c == i else 0.0 for c in range(ns)]
            for i in range(ns)]
    lp = linprog(c=[0.0] * ns + [-1.0] * ns, A_ub=a_ub, b_ub=[0.0] * ns, A_eq=a_eq,
                 b_eq=[0.0] * len(cols), bounds=[(0, None)] * ns + [(0, 1)] * ns, method="highs")
    if lp.status != 0:
        raise RuntimeError(f"consistency LP failed: {lp.message}")
    conserved = [i for i in range(ns) if lp.x[ns + i] > 0.5]
    unconserved = [i for i in range(ns) if i not in conserved]

    # Mass certificate: m = B c in the exact left nullspace of N_int, m >= 0, m_conserved > 0.
    mass = None
    left = nullspace(transpose(sub), ns)
    if left and conserved:
        cert = _certify(transpose(left), conserved)
        if cert is not None:
            mass = cert[1]
    elif not conserved:
        mass = [Fraction(0)] * ns
    # Leak certificate: v with N_int v >= 0 and (N_int v)_unconserved > 0.
    leak: list[Fraction] | None = []
    if unconserved:
        cert = _certify(sub, unconserved)
        leak = None if cert is None else cert[0]
    return {**result, "consistent": not unconserved, "conserved": conserved,
            "unconserved": unconserved, "mass": mass, "leak_combination": leak,
            "certified": mass is not None and leak is not None}
