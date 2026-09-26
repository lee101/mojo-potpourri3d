"""VectorHeatMethodSolver, the Python half.

Same shape as `_heat.py`: the geometry comes from `IntrinsicGeometry` and the
two complex factorizations -- the scalar heat operator behind `extend_scalar`
and the vertex connection Laplacian behind `transport_tangent_vectors` -- are
assembled here and factored by `mojopp3d.linalg`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from ._lib import addr, lib
from ._solver import Factorization


def _unit(z: np.ndarray) -> np.ndarray:
    mag = np.abs(z)
    out = np.zeros_like(z)
    good = mag > 0
    out[good] = z[good] / mag[good]
    return out


class VectorHeatMethodSolver:
    def __init__(self, mesh, geom, t_coef: float = 1.0):
        self.mesh = mesh
        self.geom = geom
        self.t_coef = t_coef
        self.n = mesh.n_vertices

        # Compute mean edge length and set shortTime, over `mesh.edges()`.
        total = 0.5 * (geom.edge_lengths.sum() + geom.edge_lengths[mesh.is_boundary_halfedge].sum())
        self.short_time = t_coef * (total / mesh.n_edges) ** 2

        self.mass = geom.vertex_dual_areas
        self._vector_solver = None
        self._scalar_solver = None
        self._connection_laplacian = None

    # ---- operators, factored on first use (gc's ensureHave* pattern)

    def _scalar_heat_solver(self) -> Factorization:
        if self._scalar_solver is None:
            ti, tj, tv = self.geom.cotan_laplacian_triplets()
            idx = np.arange(self.n, dtype=np.int64)
            self._scalar_solver = Factorization(
                self.n,
                np.concatenate([ti, idx]),
                np.concatenate([tj, idx]),
                np.concatenate([self.short_time * tv, self.mass]),
            )
        return self._scalar_solver

    def connection_laplacian_triplets(self):
        return self.geom.vertex_connection_laplacian_triplets()

    def connection_laplacian_matrix(self) -> scipy.sparse.csr_matrix:
        """The operator `get_connection_laplacian` hands back to Python."""
        if self._connection_laplacian is None:
            ti, tj, tr, tim = self.connection_laplacian_triplets()
            self._connection_laplacian = scipy.sparse.coo_matrix(
                (tr + 1j * tim, (ti, tj)), shape=(self.n, self.n)
            ).tocsr()
        return self._connection_laplacian

    def _vector_heat_solver(self) -> Factorization:
        if self._vector_solver is None:
            ti, tj, tr, tim = self.connection_laplacian_triplets()
            idx = np.arange(self.n, dtype=np.int64)
            self._vector_solver = Factorization(
                self.n,
                np.concatenate([ti, idx]),
                np.concatenate([tj, idx]),
                np.concatenate([self.short_time * tr, self.mass]),
                np.concatenate([self.short_time * tim, np.zeros(self.n)]),
            )
        return self._vector_solver

    # ---- API

    def extend_scalar(self, v_inds, values):
        v_inds = np.ascontiguousarray(np.atleast_1d(v_inds), dtype=np.int64)
        values = np.ascontiguousarray(np.atleast_1d(values), dtype=np.float64)
        data_rhs = np.zeros(self.n, dtype=np.float64)
        indicator_rhs = np.zeros(self.n, dtype=np.float64)
        lib().mpp3d_vector_extend_scalar_rhs(
            *self.mesh.args(), addr(v_inds), addr(values), v_inds.size,
            addr(data_rhs), addr(indicator_rhs),
        )
        solver = self._scalar_heat_solver()
        return solver.solve_vector(data_rhs) / solver.solve_vector(indicator_rhs)

    def get_tangent_frames(self):
        # Stored as [basisX, basisY] per vertex, flat.
        frames = self.geom.vertex_tangent_basis.reshape(-1, 6)
        return frames[:, 0:3], frames[:, 3:6], self.geom.vertex_normals.reshape(-1, 3)

    def get_connection_laplacian(self):
        return self.connection_laplacian_matrix()

    def transport_tangent_vectors(self, v_inds, vectors):
        v_inds = np.ascontiguousarray(np.atleast_1d(v_inds), dtype=np.int64)
        vectors = np.ascontiguousarray(np.atleast_1d(vectors), dtype=np.float64)
        n_sources = v_inds.size
        rhs_re = np.zeros(self.n, dtype=np.float64)
        rhs_im = np.zeros(self.n, dtype=np.float64)
        lib().mpp3d_vector_transport_rhs(
            *self.mesh.args(), addr(v_inds), addr(vectors), n_sources,
            addr(rhs_re), addr(rhs_im),
        )
        direction = _unit(
            self._vector_heat_solver().solve(np.stack([rhs_re, rhs_im], axis=1))[:, 0]
        )

        if n_sources == 1:
            # For one source, can just normalize and project
            scale = np.linalg.norm(vectors[0])
        else:
            # For multiple sources, need to interpolate magnitudes
            scale = self.extend_scalar(v_inds, np.linalg.norm(vectors, axis=1))
        return np.stack([direction.real * scale, direction.imag * scale], axis=1)

    def transport_tangent_vector(self, v_ind, vector):
        # gc's binding returns EigenMap<double, 2>, i.e. the whole N x 2 field,
        # for the singular spelling as well.
        return self.transport_tangent_vectors([v_ind], [vector])

    def compute_log_map(self, v_ind, strategy="AffineLocal"):
        """Not covered -- see the README.

        All three strategies end in a factorization of a singular or indefinite
        operator (the affine connection heat operator, or the unshifted Poisson
        operator), which upstream hands to Eigen's SparseLU. This port only has
        a pivot-clamped LDL^T, and on those operators it returns a different
        answer rather than a wrong one being silently reported.
        """
        raise NotImplementedError(
            "compute_log_map is not covered by this port; see the README"
        )
