"""`potpourri3d.point_cloud`, with the local triangulation and the heat-method
distance solver backed by the Mojo kernels. Names, argument order and defaults
are upstream's.

`PointCloudHeatSolver` builds the same chain upstream does: the local Delaunay
triangulation of every neighbourhood (`PointCloudLocalTriangulation`), the flat
triangle list it hands back, then the same mollify / tufted-cover / flip-to-
Delaunay passes a mesh goes through for the robust Laplacian, and finally the
heat method on that cover. The cover is intrinsic data, so the solver is
constructed from edge lengths rather than from positions.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib
from ._heat import HeatMethodDistanceSolver
from ._solver import HalfedgeMesh, IntrinsicGeometry
from ._tufted import tufted_intrinsic_mesh
from .core import *

# `PointPositionGeometry::kNeighborSize`
K_NEIGHBOR_SIZE = 30

# `computeTuftedTriangulation` mollifies with this factor
MOLLIFY_FACTOR = 1e-5

_NOT_COVERED = (
    "not covered by this port; see the coverage table in the README"
)


class PointCloudLocalTriangulation:
    """`geometrycentral::pointcloud::PointCloudLocalTriangulation`."""

    def __init__(self, P, with_degeneracy_heuristic=True):
        validate_points(P)
        self.P = f64(P)
        self.with_degeneracy_heuristic = bool(with_degeneracy_heuristic)
        n = self.P.shape[0]
        # A point cloud of fewer than k+1 points has fewer neighbours than k.
        self.k = min(K_NEIGHBOR_SIZE, max(n - 1, 1))

        self.neighbors = np.zeros(n * self.k, dtype=np.int64)
        keys = np.zeros(self.k, dtype=np.float64)
        vals = np.zeros(self.k, dtype=np.int64)
        lib().mpp3d_pc_neighbors(
            addr(self.P), n, self.k, addr(self.neighbors), addr(keys), addr(vals)
        )

        self.normals = np.zeros(3 * n, dtype=np.float64)
        a = np.zeros(9, dtype=np.float64)
        v = np.zeros(9, dtype=np.float64)
        lib().mpp3d_pc_normals(
            addr(self.P), addr(self.neighbors), n, self.k, addr(self.normals),
            addr(a), addr(v),
        )

        self.tangent_coordinates = np.zeros(2 * n * self.k, dtype=np.float64)
        lib().mpp3d_pc_tangent_coordinates(
            addr(self.P), addr(self.normals), addr(self.neighbors), n, self.k,
            addr(self.tangent_coordinates),
        )

    def _build(self):
        """The flat triangle list, `handleToFlatInds`."""
        n = self.P.shape[0]
        k = self.k
        # A point's star is a fan over its angularly sorted neighbours, so it
        # has at most k triangles.
        tri = np.zeros(3 * n * k, dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        pts = np.zeros(2 * k, dtype=np.float64)
        angles = np.zeros(k, dtype=np.float64)
        sort_inds = np.zeros(k, dtype=np.int64)
        total = lib().mpp3d_pc_local_triangulation(
            addr(self.tangent_coordinates), addr(self.neighbors), n, k,
            1 if self.with_degeneracy_heuristic else 0, addr(tri), addr(offsets),
            addr(pts), addr(angles), addr(sort_inds),
        )
        return tri[:total], offsets

    def get_local_triangulation(self):
        """Return the local point cloud triangulation

        The out matrix has the following convention:
            size: num_points, max_neighs, 3. max_neighs is the maximum number of neighbors
            out[point_idx, neigh_idx, :] are the indices of the 3 neighbors
            -1 is used as the fill value for unused elements if num_neighs < max_neighs for a point
        """
        tri, offsets = self._build()
        n = self.P.shape[0]
        counts = (offsets[1:] - offsets[:-1]) // 3
        max_neighs = int(counts.max()) if n else 0
        out = np.full((n, max_neighs, 3), -1, dtype=np.int64)
        for p in range(n):
            c = int(counts[p])
            out[p, :c] = tri[offsets[p] : offsets[p] + 3 * c].reshape(c, 3)
        return out

    def flat_triangles(self):
        """`handleToFlatInds`: every point's triangles, concatenated."""
        tri, _ = self._build()
        return tri


class PointCloudHeatSolver:
    """`PointCloudHeatSolver`: heat-method distance on a point cloud."""

    def __init__(self, P, t_coef=1.0):
        validate_points(P)
        self.P = f64(P)
        self.t_coef = t_coef
        n = self.P.shape[0]

        # `requireTuftedTriangulation`: local triangulation, then the same
        # intrinsic preprocessing a mesh gets for the robust Laplacian.
        self.local_triangulation = PointCloudLocalTriangulation(
            P, with_degeneracy_heuristic=True
        )
        F = self.local_triangulation.flat_triangles().reshape(-1, 3)
        F_tufted, twins, edge_lengths, self.n_flips = tufted_intrinsic_mesh(
            F, self.P, n, MOLLIFY_FACTOR
        )
        self.mesh = HalfedgeMesh(F_tufted, n, twins)
        self.geom = IntrinsicGeometry.from_edge_lengths(self.mesh, edge_lengths)
        self.solver = HeatMethodDistanceSolver(self.mesh, self.geom, t_coef)

    def compute_distance(self, p_ind):
        return self.solver.compute_distance(p_ind)

    def compute_distance_multisource(self, p_inds):
        return self.solver.compute_distance(p_inds)

    def extend_scalar(self, p_inds, values):
        if len(p_inds) != len(values):
            raise ValueError("source point indices and values array should be same shape")
        raise NotImplementedError(_NOT_COVERED)

    def get_tangent_frames(self):
        raise NotImplementedError(_NOT_COVERED)

    def transport_tangent_vector(self, p_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        raise NotImplementedError(_NOT_COVERED)

    def transport_tangent_vectors(self, p_inds, vectors):
        if len(p_inds) != len(vectors):
            raise ValueError("source point indices and values array should be same length")
        raise NotImplementedError(_NOT_COVERED)

    def compute_log_map(self, p_ind):
        raise NotImplementedError(_NOT_COVERED)

    def compute_signed_distance(
        self, curves, cloud_normals, preserve_source_normals=False,
        level_set_constraint="ZeroSet", soft_level_set_weight=-1,
    ):
        raise NotImplementedError(_NOT_COVERED)
