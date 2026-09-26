"""HeatMethodDistanceSolver, the Python half.

Mirrors `HeatMethodDistanceSolver`'s constructor: build and factor
`M + tCoef * L` for the heat step and `L + 1e-6 I` for the Poisson step, then
run the three-step method in `computeDistance`. The geometry it consumes comes
from `IntrinsicGeometry`; the factorization from `mojopp3d.linalg`.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, i64, lib
from ._solver import Factorization


class HeatMethodDistanceSolver:
    def __init__(self, mesh: HalfedgeMesh, geom: IntrinsicGeometry, t_coef: float = 1.0):
        self.mesh = mesh
        self.geom = geom
        self.t_coef = t_coef

        ti, tj, tv = geom.cotan_laplacian_triplets()
        n = mesh.n_vertices

        # Compute mean edge length and set shortTime. gc averages over
        # `mesh.edges()`; the halfedge buffer counts interior edges twice, so
        # the boundary sum is added back before halving.
        boundary = geom.edge_lengths[mesh.is_boundary_halfedge].sum()
        total = 0.5 * (geom.edge_lengths.sum() + boundary)
        mean_edge_length = total / mesh.n_edges
        self.short_time = t_coef * mean_edge_length * mean_edge_length

        # Heat operator: M + shortTime * L
        mass = geom.vertex_dual_areas
        idx = np.arange(n, dtype=np.int64)
        self.heat_solver = Factorization(
            n,
            np.concatenate([ti, idx]),
            np.concatenate([tj, idx]),
            np.concatenate([self.short_time * tv, mass]),
        )

        # Poisson solver.
        # NOTE: In theory, it should not be necessary to shift the Laplacian:
        # cotan-Laplace is always PSD. However, when the matrix is only positive
        # SEMIdefinite, some solvers may not work.
        self.poisson_solver = Factorization(
            n,
            np.concatenate([ti, idx]),
            np.concatenate([tj, idx]),
            np.concatenate([tv, 1.0e-6 * np.ones(n)]),
        )

    def compute_distance_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """computeDistanceRHS: diffuse, take the gradient, integrate it back."""
        heat_vec = self.heat_solver.solve_vector(rhs)
        divergence = np.zeros(self.mesh.n_vertices, dtype=np.float64)
        lib().mpp3d_heat_compute_divergence(
            *self.mesh.args(),
            addr(self.geom.halfedge_vectors_in_face),
            addr(self.geom.halfedge_cotan_weights),
            addr(heat_vec),
            addr(divergence),
        )
        return self.poisson_solver.solve_vector(divergence)

    def compute_distance(self, v_inds) -> np.ndarray:
        """computeDistance: build the RHS from the sources, then shift."""
        v_inds = np.ascontiguousarray(np.atleast_1d(v_inds), dtype=np.int64)
        rhs = np.zeros(self.mesh.n_vertices, dtype=np.float64)
        lib().mpp3d_heat_build_rhs(
            *self.mesh.args(), addr(v_inds), v_inds.size, addr(rhs)
        )
        dist_vec = self.compute_distance_rhs(rhs)

        shift = np.zeros(1, dtype=np.float64)
        lib().mpp3d_heat_shift_distance(
            *self.mesh.args(),
            addr(dist_vec),
            addr(self.geom.edge_lengths),
            addr(v_inds),
            v_inds.size,
            addr(shift),
        )
        return dist_vec + shift[0]
