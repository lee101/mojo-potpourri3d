"""The Python side of the port: buffer ownership and the solver objects that
stand in for geometry-central's `IntrinsicGeometryInterface` and its sparse
Cholesky.

NumPy owns every array. The Mojo kernels take addresses, so this module is
mostly bookkeeping: allocate a flat C-contiguous buffer per quantity, call the
kernel, read the result. `Factorization` wraps the permute/symbolic/numeric
sequence and the solve, for both the real symmetric and the complex-symmetric
case.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib

class HalfedgeMesh:
    """The connectivity arrays `SurfaceMesh` builds from a face list."""

    def __init__(self, F: np.ndarray, n_vertices: int):
        F = i64(F).ravel()
        n_faces = F.size // 3
        n_he = 3 * n_faces
        self.n_he = n_he
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.he_vertex = np.zeros(n_he, dtype=np.int64)
        self.he_next = np.zeros(n_he, dtype=np.int64)
        self.he_twin = np.zeros(n_he, dtype=np.int64)
        self.he_face = np.zeros(n_he, dtype=np.int64)
        self.v_halfedge = np.zeros(n_vertices, dtype=np.int64)
        self.f_halfedge = np.zeros(n_faces, dtype=np.int64)

        m = max(n_he, 1)
        keys = np.zeros(m, dtype=np.int64)
        vals = np.zeros(m, dtype=np.int64)
        tmpk = np.zeros(m, dtype=np.int64)
        tmpv = np.zeros(m, dtype=np.int64)
        buckets = np.zeros(256, dtype=np.int64)
        lib().mpp3d_build_halfedge_mesh(
            addr(F), n_faces, n_vertices,
            addr(self.he_vertex), addr(self.he_next), addr(self.he_twin),
            addr(self.he_face), addr(self.v_halfedge), addr(self.f_halfedge),
            addr(keys), addr(vals), addr(tmpk), addr(tmpv), addr(buckets),
        )

    @property
    def n_edges(self) -> int:
        """gc's `mesh.nEdges()`; halfedge storage counts each interior edge twice."""
        return (self.n_he + int((self.he_twin < 0).sum())) // 2

    def args(self) -> list:
        return [
            addr(self.he_vertex), addr(self.he_next), addr(self.he_twin),
            addr(self.he_face), addr(self.v_halfedge), addr(self.f_halfedge),
            self.n_he, self.n_vertices, self.n_faces,
        ]


class IntrinsicGeometry:
    """`IntrinsicGeometryInterface`, restricted to the caches the heat solvers use."""

    def __init__(self, mesh: HalfedgeMesh, positions: np.ndarray):
        self.mesh = mesh
        self.positions = f64(positions)
        n_he = mesh.n_he
        n_vertices = mesh.n_vertices
        n_faces = mesh.n_faces

        self.edge_lengths = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_edge_lengths(*mesh.args(), addr(self.positions), addr(self.edge_lengths))

        self.face_areas = np.zeros(n_faces, dtype=np.float64)
        lib().mpp3d_compute_face_areas(*mesh.args(), addr(self.edge_lengths), addr(self.face_areas))

        self.vertex_dual_areas = np.zeros(n_vertices, dtype=np.float64)
        lib().mpp3d_compute_vertex_dual_areas(*mesh.args(), addr(self.face_areas), addr(self.vertex_dual_areas))

        # Intrinsic corner angles drive the tangent frames; the embedded ones
        # drive vertex normals. A VertexPositionGeometry resolves `cornerAngles`
        # to the embedded version, so both are needed.
        self.corner_angles = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_corner_angles(*mesh.args(), addr(self.edge_lengths), addr(self.corner_angles))

        self.vertex_angle_sums = np.zeros(n_vertices, dtype=np.float64)
        lib().mpp3d_compute_vertex_angle_sums(*mesh.args(), addr(self.corner_angles), addr(self.vertex_angle_sums))

        self.corner_scaled_angles = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_corner_scaled_angles(
            *mesh.args(), addr(self.corner_angles), addr(self.vertex_angle_sums), addr(self.corner_scaled_angles)
        )

        self.halfedge_cotan_weights = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_halfedge_cotan_weights(
            *mesh.args(), addr(self.edge_lengths), addr(self.face_areas), addr(self.halfedge_cotan_weights)
        )

        self.edge_cotan_weights = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_edge_cotan_weights(
            *mesh.args(), addr(self.edge_lengths), addr(self.face_areas), addr(self.edge_cotan_weights)
        )

        self.halfedge_vectors_in_face = np.zeros(2 * n_he, dtype=np.float64)
        lib().mpp3d_compute_halfedge_vectors_in_face(
            *mesh.args(), addr(self.edge_lengths), addr(self.face_areas), addr(self.halfedge_vectors_in_face)
        )

        self.halfedge_vectors_in_vertex = np.zeros(2 * n_he, dtype=np.float64)
        lib().mpp3d_compute_halfedge_vectors_in_vertex(
            *mesh.args(), addr(self.edge_lengths), addr(self.corner_scaled_angles),
            addr(self.halfedge_vectors_in_vertex),
        )

        self.transport_vectors_along_halfedge = np.zeros(2 * n_he, dtype=np.float64)
        lib().mpp3d_compute_transport_vectors_along_halfedge(
            *mesh.args(), addr(self.halfedge_vectors_in_vertex), addr(self.transport_vectors_along_halfedge)
        )

        # Embedded quantities
        self.face_normals = np.zeros(3 * n_faces, dtype=np.float64)
        lib().mpp3d_compute_face_normals(*mesh.args(), addr(self.positions), addr(self.face_normals))

        self.embedded_corner_angles = np.zeros(n_he, dtype=np.float64)
        lib().mpp3d_compute_corner_angles_embedded(
            *mesh.args(), addr(self.positions), addr(self.embedded_corner_angles)
        )

        self.vertex_normals = np.zeros(3 * n_vertices, dtype=np.float64)
        lib().mpp3d_compute_vertex_normals(
            *mesh.args(), addr(self.face_normals), addr(self.embedded_corner_angles), addr(self.vertex_normals)
        )

        self.vertex_tangent_basis = np.zeros(6 * n_vertices, dtype=np.float64)
        lib().mpp3d_compute_vertex_tangent_basis(
            *mesh.args(), addr(self.positions), addr(self.vertex_normals),
            addr(self.halfedge_vectors_in_vertex), addr(self.vertex_tangent_basis),
        )

    def cotan_laplacian_triplets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`computeCotanLaplacian`: the operator as COO triplets."""
        mesh = self.mesh
        cap = 4 * mesh.n_he
        ti = np.zeros(cap, dtype=np.int64)
        tj = np.zeros(cap, dtype=np.int64)
        tv = np.zeros(cap, dtype=np.float64)
        n = lib().mpp3d_compute_cotan_laplacian(
            *mesh.args(), addr(self.edge_cotan_weights), addr(ti), addr(tj), addr(tv)
        )
        return ti[:n], tj[:n], tv[:n]

    def vertex_connection_laplacian_triplets(self):
        """`computeVertexConnectionLaplacian`: the complex operator as COO triplets."""
        mesh = self.mesh
        cap = 2 * mesh.n_he
        ti = np.zeros(cap, dtype=np.int64)
        tj = np.zeros(cap, dtype=np.int64)
        tr = np.zeros(cap, dtype=np.float64)
        tim = np.zeros(cap, dtype=np.float64)
        n = lib().mpp3d_compute_vertex_connection_laplacian(
            *mesh.args(), addr(self.edge_cotan_weights), addr(self.transport_vectors_along_halfedge),
            addr(ti), addr(tj), addr(tr), addr(tim),
        )
        return ti[:n], tj[:n], tr[:n], tim[:n]


class Factorization:
    """A sparse LDL^T of one symmetric (or complex Hermitian) operator."""

    def __init__(self, n: int, ti, tj, tr, ti_im=None):
        self.n = n
        self.complex = ti_im is not None
        self.ti = np.ascontiguousarray(ti, dtype=np.int64)
        self.tj = np.ascontiguousarray(tj, dtype=np.int64)
        self.tr = np.ascontiguousarray(tr, dtype=np.float64)
        self.tim = np.ascontiguousarray(ti_im, dtype=np.float64) if ti_im is not None else None

        nnz_t = self.ti.size
        # Each triplet mirrors into the transposed slot too.
        cap = 2 * nnz_t + 1
        self.Ap = np.zeros(n + 1, dtype=np.int64)
        self.Ai = np.zeros(cap, dtype=np.int64)
        self.Ax = np.zeros(cap, dtype=np.float64)
        self.Axi = np.zeros(cap, dtype=np.float64)
        self.ApNew = np.zeros(n + 1, dtype=np.int64)
        self.perm = np.zeros(n, dtype=np.int64)
        self.iperm = np.zeros(n, dtype=np.int64)
        count = np.zeros(256, dtype=np.int64)
        degree = np.zeros(n, dtype=np.int64)
        visited = np.zeros(n, dtype=np.int64)
        queue = np.zeros(n, dtype=np.int64)

        # A separate buffer when there is no imaginary part: permute_upper
        # swaps and sums the two independently.
        imag = self.tim if self.complex else np.zeros(max(nnz_t, 1), dtype=np.float64)
        nnzA = lib().mpp3d_permute_upper(
            n, addr(self.ti), addr(self.tj), addr(self.tr), addr(imag), nnz_t,
            addr(degree), addr(visited), addr(queue), addr(self.perm), addr(self.iperm),
            addr(count), addr(self.Ap), addr(self.Ai), addr(self.Ax), addr(self.Axi), addr(self.ApNew),
        )
        self.Ap = self.ApNew
        self.Ai = np.ascontiguousarray(self.Ai[:nnzA])
        self.Ax = np.ascontiguousarray(self.Ax[:nnzA])
        self.Axi = np.ascontiguousarray(self.Axi[:nnzA])

        # The row index of the factor is O(nnz(L)); the rest is O(n).
        self.S_p = np.zeros(n + 1, dtype=np.int64)
        self.R_p = np.zeros(n + 1, dtype=np.int64)
        upto = np.zeros(n, dtype=np.int64)
        U_head = np.zeros(n, dtype=np.int64)
        mark = np.zeros(n, dtype=np.int64)
        cov = np.zeros(n, dtype=np.int64)
        gather = np.zeros(n, dtype=np.int64)
        aux = np.zeros(n, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        cursor = np.zeros(n, dtype=np.int64)
        info = np.zeros(2, dtype=np.int64)
        # ldl_symbolic reports how far it got when the buffer is too small, so
        # the fill rate extrapolates the size instead of doubling blindly.
        cap = max(nnzA, 1)
        while True:
            self.S_i = np.zeros(cap, dtype=np.int64)
            self.R_i = np.zeros(cap, dtype=np.int64)
            self.R_pos = np.zeros(cap, dtype=np.int64)
            U_i = np.zeros(cap, dtype=np.int64)
            U_next = np.zeros(cap, dtype=np.int64)
            got = lib().mpp3d_ldl_symbolic(
                n, addr(self.Ap), addr(self.Ai), addr(self.S_p), addr(self.S_i),
                addr(self.R_p), addr(self.R_i), addr(self.R_pos),
                addr(U_head), addr(U_i), addr(U_next), addr(mark), addr(cov),
                addr(gather), addr(aux), addr(cnt), addr(cursor), addr(info), cap,
            )
            if got >= 0:
                nnzL = got
                break
            done = max(int(info[0]), 1)
            cols = max(int(info[1]), 1)
            cap = max((2 * done * n) // cols, cap + 1)
        self.Lx = np.zeros(nnzL, dtype=np.float64)
        self.D = np.zeros(n, dtype=np.float64)
        self.Y = np.zeros(n, dtype=np.float64)
        # W is indexed from the start of a row of L, so the widest row bounds
        # it -- not the whole factor.
        self.W = np.zeros(max(int((self.R_p[1:] - self.R_p[:-1]).max()), 1), dtype=np.float64)
        if self.complex:
            self.Lxi = np.zeros(nnzL, dtype=np.float64)
            self.Di = np.zeros(n, dtype=np.float64)
            self.Yi = np.zeros(n, dtype=np.float64)
            self.Wi = np.zeros_like(self.W)
            lib().mpp3d_ldl_numeric_c(
                n, nnzL, addr(self.Ap), addr(self.Ai), addr(self.Ax), addr(self.Axi),
                addr(self.S_p), addr(self.S_i), addr(self.R_p), addr(self.R_i), addr(self.R_pos), addr(upto),
                addr(self.Lx), addr(self.Lxi), addr(self.D), addr(self.Di),
                addr(self.Y), addr(self.Yi), addr(self.W), addr(self.Wi),
            )
        else:
            lib().mpp3d_ldl_numeric(
                n, nnzL, addr(self.Ap), addr(self.Ai), addr(self.Ax),
                addr(self.S_p), addr(self.S_i), addr(self.R_p), addr(self.R_i), addr(self.R_pos), addr(upto),
                addr(self.Lx), addr(self.D), addr(self.Y), addr(self.W),
            )

    def solve(self, B: np.ndarray) -> np.ndarray:
        """Solve A X = B, undoing the permutation.

        B is `n x nrhs`. For a complex-symmetric operator, `nrhs` is 2 and the
        two columns are the real and imaginary parts; the result is complex.
        """
        n = self.n
        Bm = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
        if Bm.ndim == 1:
            Bm = Bm[:, None]
        if self.complex and Bm.shape[1] != 2:
            raise ValueError(
                "a complex operator needs the real and imaginary parts of the "
                "right-hand side as two columns"
            )
        nrhs = Bm.shape[1]
        Bp = np.ascontiguousarray(Bm[self.perm, :])
        Xp = np.zeros((n, nrhs), dtype=np.float64)
        if self.complex:
            # Column 0 of Bm is the real part of the right-hand side and column
            # 1 the imaginary part; the kernel takes them as two buffers.
            Bp_re = np.ascontiguousarray(Bp[:, 0])
            Bp_im = np.ascontiguousarray(Bp[:, 1])
            Xp_re = np.zeros(n, dtype=np.float64)
            Xp_im = np.zeros(n, dtype=np.float64)
            lib().mpp3d_ldl_solve_c(
                n, addr(self.S_p), addr(self.S_i), addr(self.Lx), addr(self.Lxi),
                addr(self.D), addr(self.Di), addr(Bp_re), addr(Bp_im),
                addr(Xp_re), addr(Xp_im), 1,
            )
            X = np.zeros((n, 1), dtype=np.complex128)
            X[self.perm, 0] = Xp_re + 1j * Xp_im
            return X
        lib().mpp3d_ldl_solve(
            n, addr(self.S_p), addr(self.S_i), addr(self.Lx), addr(self.D), addr(Bp), addr(Xp), nrhs
        )
        X = np.zeros((n, nrhs), dtype=np.float64)
        X[self.perm, :] = Xp
        return X

    def solve_vector(self, b: np.ndarray) -> np.ndarray:
        return self.solve(np.asarray(b, dtype=np.float64).reshape(-1, 1))[:, 0]

    def solve_real_rhs(self, b: np.ndarray) -> np.ndarray:
        """Solve A x = b for a real right-hand side against a complex operator."""
        if not self.complex:
            return self.solve_vector(b)
        n = self.n
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(-1))
        b_re = np.ascontiguousarray(b[self.perm])
        b_im = np.zeros(n, dtype=np.float64)
        x_re = np.zeros(n, dtype=np.float64)
        x_im = np.zeros(n, dtype=np.float64)
        lib().mpp3d_ldl_solve_c(
            n, addr(self.S_p), addr(self.S_i), addr(self.Lx), addr(self.Lxi),
            addr(self.D), addr(self.Di), addr(b_re), addr(b_im), addr(x_re), addr(x_im), 1,
        )
        x = np.zeros(n, dtype=np.complex128)
        x[self.perm] = x_re + 1j * x_im
        return x
