"""`potpourri3d.mesh`, with the heat-method and vector-heat solvers backed by
the Mojo kernels. Names, argument order and defaults are upstream's.
"""

import numpy as np
import scipy.sparse

from ._lib import addr, f64, i64, lib
from ._heat import HeatMethodDistanceSolver
from ._solver import HalfedgeMesh, IntrinsicGeometry
from ._tufted import tufted_intrinsic_mesh
from ._vector_heat import VectorHeatMethodSolver
from .core import *


class MeshHeatMethodDistanceSolver:
    """`HeatMethodDistanceSolver`, with the robust Laplacian.

    `useRobustLaplacian` runs the mesh through `buildIntrinsicTuftedCover`,
    `mollifyIntrinsic` and `flipToDelaunay` first (Sharp & Crane, SGP 2020),
    and every quantity below is then read off the tufted cover instead of the
    input mesh. The cover is intrinsic data, so it is consumed as an
    `EdgeLengthGeometry` rather than from positions.
    """

    def __init__(self, V, F, t_coef=1.0, use_robust=True):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.V = f64(V)
        self.F = i64(F)
        self.t_coef = t_coef
        self.use_robust = use_robust
        if use_robust:
            F_tufted, twins, edge_lengths, self.n_flips = tufted_intrinsic_mesh(
                self.F, self.V, self.V.shape[0]
            )
            self.mesh = HalfedgeMesh(F_tufted, self.V.shape[0], twins)
            self.geom = IntrinsicGeometry.from_edge_lengths(self.mesh, edge_lengths)
        else:
            self.mesh = HalfedgeMesh(self.F, self.V.shape[0])
            self.geom = IntrinsicGeometry(self.mesh, self.V)
        self.solver = HeatMethodDistanceSolver(self.mesh, self.geom, t_coef)

    def compute_distance(self, v_ind):
        return self.solver.compute_distance(v_ind)

    def compute_distance_multisource(self, v_inds):
        return self.solver.compute_distance(v_inds)


def compute_distance(V, F, v_ind):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance(v_ind)


def compute_distance_multisource(V, F, v_inds):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance_multisource(v_inds)


class MeshVectorHeatSolver:
    def __init__(self, V, F, t_coef=1.0, use_intrinsic_delaunay=True):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.V = f64(V)
        self.F = i64(F)
        self.t_coef = t_coef
        self.use_intrinsic_delaunay = use_intrinsic_delaunay
        self.mesh = HalfedgeMesh(self.F, self.V.shape[0])
        self.geom = IntrinsicGeometry(self.mesh, self.V)
        self.solver = VectorHeatMethodSolver(self.mesh, self.geom, t_coef)

    def extend_scalar(self, v_inds, values):
        if len(v_inds) != len(values):
            raise ValueError("source vertex indices and values array should be same shape")
        return self.solver.extend_scalar(v_inds, values)

    def get_tangent_frames(self):
        return self.solver.get_tangent_frames()

    def get_connection_laplacian(self):
        return self.solver.get_connection_laplacian()

    def transport_tangent_vector(self, v_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        return self.solver.transport_tangent_vector(v_ind, vector)

    def transport_tangent_vectors(self, v_inds, vectors):
        if len(v_inds) != len(vectors):
            raise ValueError("source vertex indices and values array should be same length")
        return self.solver.transport_tangent_vectors(v_inds, vectors)

    def compute_log_map(self, v_ind, strategy="AffineLocal"):
        return self.solver.compute_log_map(v_ind, strategy)


def cotan_laplacian(V, F, denom_eps=0.0):
    validate_mesh(V, F, force_triangular=True)
    nV = V.shape[0]
    nF = F.shape[0]

    n = 12 * nF
    mat_i = np.zeros(n, dtype=np.int64)
    mat_j = np.zeros(n, dtype=np.int64)
    mat_data = np.zeros(n, dtype=np.float64)
    nnz = lib().mpp3d_cotan_laplacian_triplets(
        addr(f64(V)), addr(i64(F)), nF, denom_eps, addr(mat_i), addr(mat_j), addr(mat_data)
    )

    L_coo = scipy.sparse.coo_matrix(
        (mat_data[:nnz], (mat_i[:nnz], mat_j[:nnz])), shape=(nV, nV)
    )

    return L_coo.tocsr()


def face_areas(V, F):
    validate_mesh(V, F, force_triangular=True)
    nF = F.shape[0]
    out = np.zeros(nF, dtype=np.float64)
    lib().mpp3d_face_areas(addr(f64(V)), addr(i64(F)), nF, addr(out))
    return out


def vertex_areas(V, F):
    validate_mesh(V, F, force_triangular=True)
    nF = F.shape[0]
    nV = V.shape[0]
    scratch = np.zeros(nF, dtype=np.float64)
    out = np.zeros(nV, dtype=np.float64)
    lib().mpp3d_vertex_areas(
        addr(f64(V)), addr(i64(F)), nF, nV, addr(scratch), addr(out)
    )
    return out
