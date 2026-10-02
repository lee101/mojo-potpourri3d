"""ctypes bridge to the point-cloud kernels in dist/libmojo-potpourri3d.so.

Every layout number comes from the kernel (`pp3d_pc_layout`), never from a formula repeated here, so
the two cannot drift apart. The symbol table lives here rather than in `_lib.py` because that module
belongs to the mesh slice.
"""

from __future__ import annotations

import ctypes

import numpy as np

from ._lib import I, P, addr, f64, i64, lib

# name -> (argtypes, restype). Mirrors the @export wrappers in src/ported.mojo.
_SIGNATURES: dict[str, tuple[list, object]] = {
    "pp3d_pc_layout": ([I, I, P], I),
    "pp3d_pc_prepare": ([I, I, I, P, P, P], I),
}

# Field names of PointCloudLayout in src/point_position_geometry.mojo, in the order the kernel
# writes them.
_GEOMETRY_FIELDS = [
    "w_size", "m_size", "n_f_max", "w_normals", "w_basis", "w_tan", "m_nbr", "m_tri_off", "m_tri",
]

# Slots the prepare call fills in the float64 arena's header. The values the Python side reads are
# n_tri (slot 6) and max_neighs (slot 7); slots 2-5 and 8-15 are reserved so the int64 fields can
# follow the float64 ones without an offset table.
_HEADER_N_TRI = 6
_HEADER_MAX_NEIHS = 7

_bound: ctypes.CDLL | None = None


def dll() -> ctypes.CDLL:
    """The shared library with the point-cloud entry points bound."""
    global _bound
    if _bound is not None:
        return _bound
    d = lib()
    for name, (argtypes, restype) in _SIGNATURES.items():
        try:
            fn = getattr(d, name)
        except AttributeError as exc:  # pragma: no cover - depends on the build
            raise RuntimeError(
                f"{name} is missing from the shared library; the point-cloud exports have to be "
                "wired into src/ported.mojo"
            ) from exc
        fn.argtypes = argtypes
        fn.restype = restype
    _bound = d
    return _bound


def geometry_layout(n_points: int, n_neighbors: int = 30) -> dict[str, int]:
    """The point-geometry arena layout, straight from the kernel."""
    out = np.zeros(9, dtype=np.int64)
    rc = dll().pp3d_pc_layout(n_points, n_neighbors, out.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"point cloud layout failed with code {rc}")
    return dict(zip(_GEOMETRY_FIELDS, (int(x) for x in out)))


__all__ = ["addr", "dll", "f64", "geometry_layout", "i64", "n_triangles_of", "max_neighs_of"]


def n_triangles_of(w: np.ndarray) -> int:
    """The triangle count pp3d_pc_prepare reported in the arena header."""
    return int(w[_HEADER_N_TRI])


def max_neighs_of(w: np.ndarray) -> int:
    """The largest per-point triangle count, i.e. the fan width upstream pads its output to."""
    return int(w[_HEADER_MAX_NEIHS])
