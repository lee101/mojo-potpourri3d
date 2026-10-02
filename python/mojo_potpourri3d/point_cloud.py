"""potpourri3d.point_cloud, with the compute served by the Mojo kernels.

Names, argument order, defaults and the ValueError checks match upstream, so this is a drop-in
replacement for the covered subset.

Covered: `PointCloudLocalTriangulation.get_local_triangulation` and
`PointCloudHeatSolver.get_tangent_frames`. Names, argument order, defaults and the ValueError
checks match upstream.
"""

from __future__ import annotations

import numpy as np

from .core import *
from ._lib_pointcloud import addr, dll, f64, geometry_layout, i64, max_neighs_of, n_triangles_of

# PointPositionGeometry::kNeighborSize
K_NEIGHBOR_SIZE = 30

__all__ = ["K_NEIGHBOR_SIZE", "PointCloudHeatSolver", "PointCloudLocalTriangulation"]


class _Geometry:
    """PointPositionGeometry's quantities: neighbours, frames, tangent coordinates, triangulation."""

    def __init__(self, P, with_degeneracy_heuristic=True):
        validate_points(P)
        P = f64(P)
        self.P = P
        self.n_points = int(P.shape[0])
        self.n_neighbors = K_NEIGHBOR_SIZE
        self.with_degeneracy_heuristic = bool(with_degeneracy_heuristic)

        lay = geometry_layout(self.n_points, self.n_neighbors)
        self._lay = lay
        self._W = np.zeros(lay["w_size"], dtype=np.float64)
        self._M = np.zeros(lay["m_size"], dtype=np.int64)
        rc = dll().pp3d_pc_prepare(
            self.n_points, self.n_neighbors, int(self.with_degeneracy_heuristic),
            addr(P), addr(self._W), addr(self._M))
        if rc != 0:
            raise RuntimeError(f"point cloud geometry setup failed with code {rc}")

        self.n_triangles = n_triangles_of(self._W)
        self.max_neighs = max_neighs_of(self._W)

    def local_triangles(self):
        """The local triangulation as a flat (n_triangles, 3) array of point indices."""
        o = self._lay["m_tri"]
        tris = self._M[o:o + 3 * self.n_triangles]
        return tris.reshape(self.n_triangles, 3).astype(np.int64)

    def triangle_counts(self):
        o = self._lay["m_tri_off"]
        return np.diff(self._M[o:o + self.n_points + 1]).astype(np.int64)


class PointCloudHeatSolver:
    """potpourri3d.PointCloudHeatSolver.

    Only `get_tangent_frames` is covered, because it is the only entry point that does not need the
    heat operator, and every heat operator on a point cloud goes through a mesh this port cannot
    build. Upstream turns the local triangulation into a geometry-central *general* SurfaceMesh, whose
    edges carry more than two halfedges: on a 300-point sphere, 794 of its 927 edges carry six. The
    intrinsic tufted cover that the robust heat method needs is then built with `separateToNewEdge`,
    which that mesh type has and this port's manifold-only corner array (one twin pair per edge, a
    self-twin for a boundary edge) cannot represent. Rather than return a field that is one to twenty
    percent away from upstream's, the methods that need the operator raise. See the coverage section
    of the README.
    """

    _NEEDS_OPERATOR = (
        "needs the point cloud's intrinsic tufted cover, which this port does not build; see the "
        "coverage section of the README"
    )

    def __init__(self, P, t_coef=1.0):
        self._geom = _Geometry(P, with_degeneracy_heuristic=True)
        self.n_points = self._geom.n_points
        self.t_coef = float(t_coef)

    def get_tangent_frames(self):
        """(basisX, basisY, normal) per point, each an (n_points, 3) array, as upstream returns."""
        o = self._geom._lay["w_normals"]
        normals = self._geom._W[o:o + 3 * self.n_points].reshape(self.n_points, 3).copy()
        o = self._geom._lay["w_basis"]
        basis = self._geom._W[o:o + 6 * self.n_points].reshape(self.n_points, 6)
        return basis[:, :3].copy(), basis[:, 3:].copy(), normals

    def compute_distance(self, p_ind):
        raise NotImplementedError(f"compute_distance {self._NEEDS_OPERATOR}")

    def compute_distance_multisource(self, p_inds):
        raise NotImplementedError(f"compute_distance_multisource {self._NEEDS_OPERATOR}")

    def extend_scalar(self, p_inds, values):
        if len(p_inds) != len(values):
            raise ValueError("source point indices and values array should be same shape")
        raise NotImplementedError(f"extend_scalar {self._NEEDS_OPERATOR}")

    def transport_tangent_vector(self, p_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        raise NotImplementedError(f"transport_tangent_vector {self._NEEDS_OPERATOR}")

    def transport_tangent_vectors(self, p_inds, vectors):
        if len(p_inds) != len(vectors):
            raise ValueError("source point indices and values array should be same length")
        raise NotImplementedError(f"transport_tangent_vectors {self._NEEDS_OPERATOR}")

    def compute_log_map(self, p_ind):
        raise NotImplementedError(f"compute_log_map {self._NEEDS_OPERATOR}")

    def compute_signed_distance(self, curves, cloud_normals, preserve_source_normals=False,
                                level_set_constraint="ZeroSet", soft_level_set_weight=-1):
        raise NotImplementedError(
            "compute_signed_distance needs a level-set solve on the sign function, which is out of "
            "scope for this port; see the coverage section of the README"
        )


class PointCloudLocalTriangulation:
    """potpourri3d.PointCloudLocalTriangulation."""

    def __init__(self, P, with_degeneracy_heuristic=True):
        self._geom = _Geometry(P, with_degeneracy_heuristic=with_degeneracy_heuristic)

    def get_local_triangulation(self):
        """Return the local point cloud triangulation

        The out matrix has the following convention:
            size: num_points, max_neighs, 3. max_neighs is the maximum number of neighbors
            out[point_idx, neigh_idx, :] are the indices of the 3 neighbors
            -1 is used as the fill value for unused elements if num_neighs < max_neighs for a point
        """
        tris = self._geom.local_triangles()
        counts = self._geom.triangle_counts()
        n = self._geom.n_points
        max_neighs = int(counts.max()) if counts.size else 0
        out = np.full((n, max_neighs, 3), -1, dtype=np.int64)
        if max_neighs == 0 or tris.shape[0] == 0:
            return out
        # The ragged per-point run lengths become a gather index, so the kernel's flat triangle
        # list is scattered once and the -1 padding is written by np.full rather than by a Python
        # loop over the points.
        rows = np.repeat(np.arange(n, dtype=np.int64), counts)
        cols = _within(counts)
        out[rows, cols] = tris
        return out


def _within(counts):
    """Concatenated arange(c) for each c, the position of each triangle within its point."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    return np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
