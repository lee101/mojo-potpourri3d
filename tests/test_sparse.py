"""The sparse symmetric LDL^T behind every solver here, against SciPy.

The heat method is only as good as this, and the heat method is asserted against upstream
everywhere else, so this file pins the factorisation and the solve on their own. The complex
(`w == 2`) path, which the vector heat method's connection Laplacian needs, is not covered: see the
coverage section of the README.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

I, F, P = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
LIB = str(Path(__file__).resolve().parents[1] / "dist" / "libmojo-potpourri3d.so")


@pytest.fixture(scope="module")
def chol():
    dll = ctypes.CDLL(LIB)
    dll.pp3d_chol_analyze.argtypes = [I, I, P, P, P, P]
    dll.pp3d_chol_analyze.restype = I
    dll.pp3d_chol_factorize.argtypes = [I, I] + [P] * 8 + [I, P, P]
    dll.pp3d_chol_factorize.restype = I
    dll.pp3d_chol_solve.argtypes = [I, I] + [P] * 7
    dll.pp3d_chol_solve.restype = I
    return dll


def factor_and_solve(dll, A, b, w=1):
    """Factor the upper triangle of the symmetric A and solve A x = b."""
    n = A.shape[0]
    upper = sp.triu(sp.csc_matrix(A), format="csc")
    upper.sort_indices()
    ap = np.ascontiguousarray(upper.indptr.astype(np.int64))
    ai = np.ascontiguousarray(upper.indices.astype(np.int64))
    ax = np.ascontiguousarray(upper.data.astype(np.float64))
    perm = np.zeros(n, dtype=np.int64)
    nnz = dll.pp3d_chol_analyze(I(n), I(w), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data,
                                 perm.ctypes.data)
    assert nnz > 0
    li = np.zeros(n + 1 + nnz, dtype=np.int64)
    lf = np.zeros(nnz + n, dtype=np.float64)
    rc = dll.pp3d_chol_factorize(
        I(n), I(w), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data, perm.ctypes.data,
        li.ctypes.data, int(li.ctypes.data + (n + 1) * 8), lf.ctypes.data,
        int(lf.ctypes.data + nnz * 8), I(nnz), P(0), P(0))
    assert rc == 0
    b = np.ascontiguousarray(b, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    rc = dll.pp3d_chol_solve(
        I(n), I(w), li.ctypes.data, int(li.ctypes.data + (n + 1) * 8), lf.ctypes.data,
        int(lf.ctypes.data + nnz * 8), perm.ctypes.data, b.ctypes.data, x.ctypes.data)
    assert rc == 0
    return x


def spd_sparse(n, density, seed):
    rng = np.random.default_rng(seed)
    B = sp.random(n, n, density=density, random_state=seed, format="csr")
    A = (B @ B.T).toarray() + np.eye(n) * 0.1
    return A


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("rhs", ["ones", "random", "single_entry"])
def test_real_symmetric_solve_matches_scipy(chol, seed, rhs):
    n = 40 + 7 * seed
    A = spd_sparse(n, 0.12, seed)
    if rhs == "ones":
        b = np.ones(n)
    elif rhs == "random":
        b = np.random.default_rng(seed).standard_normal(n)
    else:
        b = np.zeros(n)
        b[seed * 5 % n] = 3.0
    x = factor_and_solve(chol, A, b)
    ref = spl.spsolve(sp.csc_matrix(A), b)
    assert np.abs(x - ref).max() < 1e-13 * max(1.0, np.abs(ref).max())


def _csc(A):
    upper = sp.triu(sp.csc_matrix(A), format="csc")
    upper.sort_indices()
    return (np.ascontiguousarray(upper.indptr.astype(np.int64)),
            np.ascontiguousarray(upper.indices.astype(np.int64)),
            np.ascontiguousarray(upper.data.astype(np.float64)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_borrowing_the_pattern_of_an_identically_patterned_matrix(chol, seed):
    """The heat method factorizes two operators that share a pattern, borrowing the first's.

    `pp3d_chol_factorize` takes the pattern of L from reuse_lp_addr/reuse_li_addr when they are
    nonzero. This pins that the borrowed path gives the same factor as recomputing it, which is what
    `heat_factor` relies on when it hands the heat operator's pattern to the Poisson operator's.
    """
    n = 40 + 7 * seed
    A = spd_sparse(n, 0.12, seed)
    ap, ai, ax = _csc(A)
    # A second SPD matrix with the SAME sparsity pattern but different values: the diagonal moved.
    ap2, ai2, ax2 = _csc(A + np.diag(np.linspace(0.5, 2.0, n)))

    perm = np.zeros(n, dtype=np.int64)
    nnz = chol.pp3d_chol_analyze(I(n), I(1), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data,
                                 perm.ctypes.data)
    assert nnz > 0

    def factorize(a_p, a_i, a_x, reuse=None):
        li = np.zeros(n + 1 + nnz, dtype=np.int64)
        lf = np.zeros(nnz + n, dtype=np.float64)
        rlp = reuse[0].ctypes.data if reuse else 0
        rli = reuse[1].ctypes.data if reuse else 0
        rc = chol.pp3d_chol_factorize(
            I(n), I(1), a_p.ctypes.data, a_i.ctypes.data, a_x.ctypes.data, perm.ctypes.data,
            li.ctypes.data, int(li.ctypes.data + (n + 1) * 8), lf.ctypes.data,
            int(lf.ctypes.data + nnz * 8), I(nnz), P(rlp), P(rli))
        assert rc == 0
        return li, lf

    # The first factorization computes the pattern and writes it to li.
    li, lf = factorize(ap, ai, ax)
    lp_out = li[:n + 1].copy()
    li_out = li[n + 1:].copy()
    # lp[n] is the number of entries the borrowed pattern carries, and the row indices must all be
    # in range, or `_read_pattern` would refuse it and silently fall back.
    assert lp_out[0] == 0 and lp_out[-1] == nnz
    assert li_out.min() >= 1 and li_out.max() < n

    # factorize the second (diagonal-shifted) operator both with and without the borrow
    li_b, lf_b = factorize(ap2, ai2, ax2, reuse=(lp_out, li_out))
    li_r, lf_r = factorize(ap2, ai2, ax2)
    assert np.array_equal(li_b[:n + 1], li_r[:n + 1])
    assert np.array_equal(li_b[n + 1:], li_r[n + 1:])
    assert np.abs(lf_b - lf_r).max() < 1e-14 * max(1.0, np.abs(lf_r).max())


def test_a_malformed_borrowed_pattern_falls_back_to_computing_it(chol):
    """A borrowed pattern that does not hold is refused and recomputed, not trusted."""
    n = 30
    A = spd_sparse(n, 0.15, 5)
    ap, ai, ax = _csc(A)
    perm = np.zeros(n, dtype=np.int64)
    nnz = chol.pp3d_chol_analyze(I(n), I(1), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data,
                                 perm.ctypes.data)
    assert nnz > 0
    bad_lp = np.zeros(n + 1, dtype=np.int64)   # lp[0] == 0 but lp[n] == 0 too -> refused
    bad_li = np.zeros(1, dtype=np.int64)
    li = np.zeros(n + 1 + nnz, dtype=np.int64)
    lf = np.zeros(nnz + n, dtype=np.float64)
    rc = chol.pp3d_chol_factorize(
        I(n), I(1), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data, perm.ctypes.data,
        li.ctypes.data, int(li.ctypes.data + (n + 1) * 8), lf.ctypes.data,
        int(lf.ctypes.data + nnz * 8), I(nnz), P(bad_lp.ctypes.data), P(bad_li.ctypes.data))
    assert rc == 0
    # The fallback computed the real pattern, so it matches the non-borrowed run exactly.
    li2 = np.zeros(n + 1 + nnz, dtype=np.int64)
    lf2 = np.zeros(nnz + n, dtype=np.float64)
    rc = chol.pp3d_chol_factorize(
        I(n), I(1), ap.ctypes.data, ai.ctypes.data, ax.ctypes.data, perm.ctypes.data,
        li2.ctypes.data, int(li2.ctypes.data + (n + 1) * 8), lf2.ctypes.data,
        int(lf2.ctypes.data + nnz * 8), I(nnz), P(0), P(0))
    assert rc == 0
    assert np.array_equal(li, li2)
    assert np.abs(lf - lf2).max() < 1e-14
