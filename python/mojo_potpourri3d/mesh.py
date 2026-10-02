"""potpourri3d.mesh, with the compute-bound pieces served by the Mojo kernels.

Names, argument order and defaults match upstream, so this is a drop-in replacement for the
covered subset.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from .core import *
from ._lib import addr, edge_counts, f64, heat_layout, i64, lib

# The upstream module's public surface; the ctypes helpers stay private.
__all__ = [
    "MeshHeatMethodDistanceSolver",
    "compute_distance",
    "compute_distance_multisource",
    "cotan_laplacian",
    "edges",
    "face_areas",
    "validate_mesh",
    "validate_points",
    "vertex_areas",
]


# rc == -2 is the kernel's out-of-range face index check. Upstream relies on the caller for this and
# its own check (np.amin) only catches negative indices, so the kernels check it and report it here
# with upstream's message.
def _check_rc(rc, name):
    if rc == -2:
        raise ValueError(
            "There is an out-of-bounds face index. Faces should be zero-based array of indices to "
            "vertices"
        )
    if rc != 0:
        raise RuntimeError(f"{name} failed with code {rc}")


def _check_query_rc(rc):
    """A rejected query, with the reason named.

    Upstream indexes its vertex array with the caller's source list without checking it, so an
    out-of-range index reads whatever is next in memory. Dropping the bad index here instead would
    leave an all-zero right-hand side and return a constant distance field, which is a finite and
    wrong number rather than a visible failure; both cases are reported instead.
    """
    if rc == -3:
        raise IndexError(
            "source vertex index is out of range for this mesh; it indexes the vertex array "
            "directly, so check it against V.shape[0]"
        )
    if rc == -4:
        raise ValueError("at least one source vertex is required to compute a distance")
    if rc != 0:
        raise RuntimeError(f"heat method solve failed with code {rc}")


class MeshHeatMethodDistanceSolver:
    """potpourri3d.MeshHeatMethodDistanceSolver.

    Upstream factors the heat and Poisson operators once in the constructor and reuses them; here
    the factorization lives in arenas the constructor allocates, and each query is one FFI call.
    """

    def __init__(self, V, F, t_coef=1.0, use_robust=True):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        V = f64(V)
        F = i64(F)
        self.V = V
        self.F = F
        self.t_coef = float(t_coef)
        self.use_robust = bool(use_robust)

        n_v, n_f = int(V.shape[0]), int(F.shape[0])
        n_e, _n_b = edge_counts(n_v, F)
        n_fc = 2 * n_f if self.use_robust else n_f
        n_ec = (3 * n_f + _n_b) if self.use_robust else n_e
        lay = heat_layout(n_v, n_f, n_fc, n_ec)
        self._W = np.zeros(lay[0], dtype=np.float64)
        self._M = np.zeros(lay[1], dtype=np.int64)
        rc = lib().pp3d_heat_setup(
            n_v, n_f, addr(V), addr(F), self.t_coef, int(self.use_robust), addr(self._W),
            addr(self._M), 0,
        )
        if rc == -2:
            _check_rc(rc, "heat method setup")
        if rc == -50:
            raise ValueError(
                "the mesh has a degenerate (zero-area) face, so its cotangent weights are not "
                "finite; the heat method cannot differentiate it away. Check the face indices and "
                "the vertex positions. See the coverage section of the README."
            )
        if rc != 0:
            raise RuntimeError(f"heat method setup failed with code {rc}")

        # Each operator gets its own factor pair, so no packing and no offsets.
        nnzl_h = int(self._W[4])
        nnzl_p = int(self._W[5])
        self._n_fc = int(self._W[2])
        self._n_ec = int(self._W[9])
        li_n = n_v + 1 + max(nnzl_h, nnzl_p)
        lf_n = max(nnzl_h, nnzl_p) + n_v
        self._LI = np.zeros(2 * li_n, dtype=np.int64)
        self._LF = np.zeros(2 * lf_n, dtype=np.float64)
        rc = lib().pp3d_heat_factor(n_v, n_f, self._n_fc, self._n_ec, addr(self._W), addr(self._M),
                                    addr(self._LI), addr(self._LF), li_n, lf_n)
        if rc != 0:
            raise RuntimeError(f"heat method factorization failed with code {rc}")
        self._n_v = n_v
        self._n_f = n_f

    def compute_distance(self, v_ind):
        return self.compute_distance_multisource([v_ind])

    def compute_distance_multisource(self, v_inds):
        srcs = i64(np.atleast_1d(np.asarray(v_inds, dtype=np.int64)))
        out = np.zeros(self._n_v, dtype=np.float64)
        li_n = len(self._LI) // 2
        lf_n = len(self._LF) // 2
        rc = lib().pp3d_heat_compute_distance(
            self._n_v, self._n_f, self._n_fc, self._n_ec, addr(self._W), addr(self._M), addr(self._LI),
            addr(self._LF), addr(self._LI) + li_n * 8, addr(self._LF) + lf_n * 8, addr(srcs),
            int(srcs.shape[0]), addr(out))
        _check_query_rc(rc)
        return out


def compute_distance(V, F, v_ind):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance(v_ind)


def compute_distance_multisource(V, F, v_inds):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance_multisource(v_inds)


def cotan_laplacian(V, F, denom_eps=0.0):
    validate_mesh(V, F, force_triangular=True)
    nV = V.shape[0]
    nF = F.shape[0]
    V = f64(V)
    F = i64(F)

    mat_i = np.zeros(12 * nF, dtype=np.int64)
    mat_j = np.zeros(12 * nF, dtype=np.int64)
    mat_data = np.zeros(12 * nF, dtype=np.float64)
    rc = lib().pp3d_cotan_laplacian(
        int(nV), int(nF), addr(V), addr(F), float(denom_eps), addr(mat_i), addr(mat_j), addr(mat_data))
    _check_rc(rc, "cotan_laplacian")

    L_coo = scipy.sparse.coo_matrix((mat_data, (mat_i, mat_j)), shape=(nV, nV))
    return L_coo.tocsr()


def face_areas(V, F):
    validate_mesh(V, F, force_triangular=True)
    V = f64(V)
    F = i64(F)
    out = np.zeros(F.shape[0], dtype=np.float64)
    rc = lib().pp3d_face_areas(int(V.shape[0]), int(F.shape[0]), addr(V), addr(F), addr(out))
    _check_rc(rc, "face_areas")
    return out


def vertex_areas(V, F):
    validate_mesh(V, F, force_triangular=True)
    V = f64(V)
    F = i64(F)
    out = np.zeros(V.shape[0], dtype=np.float64)
    rc = lib().pp3d_vertex_areas(int(V.shape[0]), int(F.shape[0]), addr(V), addr(F), addr(out))
    _check_rc(rc, "vertex_areas")
    return out


def edges(V, F):
    validate_mesh(V, F, force_triangular=False)
    V = f64(V)
    F = i64(F)
    buf = np.zeros(2 * (3 * F.shape[0]), dtype=np.int64)
    n = lib().pp3d_edges(int(V.shape[0]), int(F.shape[0]), addr(F), addr(buf))
    if n < 0:
        _check_rc(n, "edges")
    return buf[: 2 * n].reshape(n, 2)
