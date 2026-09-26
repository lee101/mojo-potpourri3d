"""The general mesh and the tufted-cover passes, Python side.

`SurfaceMesh` in geometry-central's general (explicit twin) form, plus
`mollifyIntrinsic`, `buildIntrinsicTuftedCover` and `flipToDelaunay`. Both the
robust Laplacian on a triangle mesh and the whole point-cloud pipeline go
through here.

NumPy owns every array. The element counts are upper-bounded by the calls that
can grow them -- the cover at most doubles the faces and triples the halfedges,
and the flips create nothing -- so the buffers are sized once here and the fill
counts travel back and forth in three `Int64` cells that Mojo reads and writes.
"""

from __future__ import annotations

import ctypes

import numpy as np

from ._lib import addr, f64, i64, lib

INVALID = -1


class GeneralMesh:
    """`SurfaceMesh` in general mode, backed by NumPy arrays."""

    def __init__(self, F: np.ndarray, n_vertices: int):
        F = i64(F).ravel()
        n_faces = F.size // 3
        # `buildIntrinsicTuftedCover` at most doubles the faces and triples the
        # halfedges; the flips create no new elements. Six of everything leaves
        # room for the edge count, which the cover can push past 3 * n_faces.
        self.cap_faces = 2 * n_faces + 1
        self.cap_he = 6 * n_faces + 1
        self.cap_edges = 6 * n_faces + 1
        self.n_input_faces = n_faces

        self.he_vertex = np.zeros(self.cap_he, dtype=np.int64)
        self.he_next = np.zeros(self.cap_he, dtype=np.int64)
        self.he_face = np.zeros(self.cap_he, dtype=np.int64)
        self.he_edge = np.zeros(self.cap_he, dtype=np.int64)
        self.he_orient = np.zeros(self.cap_he, dtype=np.int64)
        self.he_sibling = np.full(self.cap_he, INVALID, dtype=np.int64)
        self.e_halfedge = np.zeros(self.cap_edges, dtype=np.int64)
        self.f_halfedge = np.zeros(self.cap_faces, dtype=np.int64)
        self.counts = np.zeros(3, dtype=np.int64)

        m = max(3 * n_faces, 1)
        self._scratch = [np.zeros(m, dtype=np.int64) for _ in range(4)]
        self._scratch.append(np.zeros(256, dtype=np.int64))

        lib().mpp3d_build_general_mesh(
            addr(F), n_faces, n_vertices,
            *self._arrays(), addr(self.counts),
            *(addr(a) for a in self._scratch),
        )

    def _arrays(self) -> list:
        return [
            addr(self.he_vertex), addr(self.he_next), addr(self.he_face),
            addr(self.he_edge), addr(self.he_orient), addr(self.he_sibling),
            addr(self.e_halfedge), addr(self.f_halfedge),
        ]

    @property
    def n_he(self) -> int:
        return int(self.counts[0])

    @property
    def n_faces(self) -> int:
        return int(self.counts[1])

    @property
    def n_edges(self) -> int:
        return int(self.counts[2])

    def faces(self) -> np.ndarray:
        F = np.zeros((self.n_faces, 3), dtype=np.int64)
        lib().mpp3d_general_write_faces(*self._arrays(), addr(self.counts), addr(F))
        return F

    def twins(self) -> np.ndarray:
        out = np.zeros(self.n_he, dtype=np.int64)
        index_of = np.zeros(self.n_he, dtype=np.int64)
        lib().mpp3d_general_write_twins(
            *self._arrays(), addr(self.counts), addr(out), addr(index_of)
        )
        return out

    def edge_lengths(self, V: np.ndarray) -> np.ndarray:
        """Per-edge lengths, from positions, for a mesh that still has them."""
        V = f64(V)
        a = V[self.he_vertex[: self.n_he]]
        b = V[self.he_vertex[self.he_next[: self.n_he]]]
        per_halfedge = np.linalg.norm(a - b, axis=1)
        # one entry per edge, read off that edge's representative halfedge. The
        # cover adds edges, so this is sized for the capacity, like gc's
        # `EdgeData`.
        out = np.zeros(self.cap_edges, dtype=np.float64)
        out[: self.n_edges] = per_halfedge[self.e_halfedge[: self.n_edges]]
        return out

    def halfedge_edge_lengths(self, edge_lengths: np.ndarray) -> np.ndarray:
        edge_lengths = f64(edge_lengths)
        out = np.zeros(self.n_he, dtype=np.float64)
        lib().mpp3d_general_write_halfedge_edge_lengths(
            *self._arrays(), addr(self.counts), addr(edge_lengths), addr(out)
        )
        return out

    def mollify_intrinsic(self, edge_lengths: np.ndarray, relative_factor: float) -> float:
        """`mollifyIntrinsic`: offset every edge length by one scalar."""
        edge_lengths = f64(edge_lengths)
        return lib().mpp3d_mollify_intrinsic(
            *self._arrays(), addr(self.counts), addr(edge_lengths), self.n_edges,
            ctypes.c_double(relative_factor),
        )

    def build_intrinsic_tufted_cover(self, edge_lengths: np.ndarray) -> int:
        """`buildIntrinsicTuftedCover`, the intrinsic-only (no positions) form."""
        edge_lengths = f64(edge_lengths)
        n_orig_faces = self.n_faces
        n_orig_edges = self.n_edges
        other_sheet = np.zeros(self.cap_he, dtype=np.int64)
        is_front = np.zeros(self.cap_faces, dtype=np.int64)
        is_orig_edge = np.zeros(self.cap_edges, dtype=np.int64)
        edge_faces = np.zeros(self.cap_he, dtype=np.int64)
        n_new = lib().mpp3d_build_intrinsic_tufted_cover(
            *self._arrays(), addr(self.counts), addr(edge_lengths),
            n_orig_faces, n_orig_edges, addr(other_sheet), addr(is_front),
            addr(is_orig_edge), addr(edge_faces), self.cap_faces, self.cap_he,
            self.cap_edges,
        )
        if n_new < 0:
            raise RuntimeError("tufted cover overflowed its buffers")
        return n_new

    def flip_to_delaunay(self, edge_lengths: np.ndarray, delaunay_eps: float = 1e-6) -> int:
        """`flipToDelaunay`: flip until no edge fails the Delaunay test."""
        edge_lengths = f64(edge_lengths)
        n_edges = self.n_edges
        # The queue starts with every edge and grows by at most four per flip.
        cap = 4 * n_edges + 4096
        queue = np.zeros(cap, dtype=np.int64)
        in_queue = np.zeros(cap, dtype=np.int64)
        n_flips = lib().mpp3d_flip_to_delaunay(
            *self._arrays(), addr(self.counts), addr(edge_lengths), n_edges,
            addr(queue), addr(in_queue), cap, ctypes.c_double(delaunay_eps),
        )
        if n_flips < 0:
            raise RuntimeError(
                "flipToDelaunay hit a boundary or non-manifold edge, or its "
                "edge queue overflowed"
            )
        return n_flips


def tufted_intrinsic_mesh(
    F: np.ndarray, V: np.ndarray, n_vertices: int, relative_factor: float = 1e-5
):
    """`mollifyIntrinsic` -> `buildIntrinsicTuftedCover` -> `flipToDelaunay`.

    Returns the face list of the tufted cover, its halfedge twins, per-halfedge
    edge lengths, and the number of flips. Everything downstream consumes the
    cover as an ordinary static mesh with a given intrinsic geometry, which is
    what `EdgeLengthGeometry` is.
    """
    mesh = GeneralMesh(F, n_vertices)
    edge_lengths = mesh.edge_lengths(V)
    mesh.mollify_intrinsic(edge_lengths, relative_factor)
    mesh.build_intrinsic_tufted_cover(edge_lengths)
    n_flips = mesh.flip_to_delaunay(edge_lengths)
    F2 = mesh.faces()
    twins = mesh.twins()
    per_halfedge = mesh.halfedge_edge_lengths(edge_lengths)
    return F2, twins, per_halfedge, n_flips
