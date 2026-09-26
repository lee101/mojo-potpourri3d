"""ctypes bridge to the fixed-ABI Mojo kernels.

The shared library is built by `build/build.sh` (`pixi run build`). Every
buffer is a NumPy array owned by the caller; Mojo receives raw addresses, so
nothing is allocated on the far side and nothing can leak.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIB = ROOT / "dist" / "libmojo-potpourri3d.so"

I = ctypes.c_int64
F = ctypes.c_double
P = ctypes.c_void_p

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list, object]] = {
    "mpp3d_build_halfedge_mesh": ([P, I, I] + [P] * 11 + [P], I),
    "mpp3d_rcm": ([I, P, P, P, P, P, P], None),
    "mpp3d_adjacency_pattern": ([I, P, P, I, P, P], I),
    "mpp3d_vector_extend_scalar_rhs": ([P] * 6 + [I] * 3 + [P, P, I, P, P], None),
    "mpp3d_vector_transport_rhs": ([P] * 6 + [I] * 3 + [P, P, I, P, P], None),
    "mpp3d_heat_build_rhs": ([P] * 6 + [I] * 3 + [P, I, P], None),
    "mpp3d_heat_compute_divergence": ([P] * 6 + [I] * 3 + [P] * 4, None),
    "mpp3d_heat_shift_distance": ([P] * 6 + [I] * 3 + [P, P, P, I, P], F),
    "mpp3d_source_face_corner": ([P] * 6 + [I] * 3 + [I, P], None),
    "mpp3d_permute_upper": ([I, P, P, P, P, I, P, P, P, P, P, P, P, P, P, P, P, P, P, P], I),
    "mpp3d_ldl_symbolic": ([I] + [P] * 17 + [I], I),
    "mpp3d_ldl_numeric": ([I, I] + [P] * 13, None),
    "mpp3d_ldl_solve": ([I, P, P, P, P, P, P, I], None),
    "mpp3d_ldl_numeric_c": ([I, I] + [P] * 18, None),
    "mpp3d_ldl_solve_c": ([I] + [P] * 10 + [I], None),
    "mpp3d_compute_edge_lengths": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_corner_angles": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_face_areas": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_vertex_dual_areas": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_vertex_angle_sums": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_corner_scaled_angles": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_halfedge_cotan_weights": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_edge_cotan_weights": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_halfedge_vectors_in_face": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_halfedge_vectors_in_vertex": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_transport_vectors_along_halfedge": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_cotan_laplacian": ([P] * 6 + [I] * 3 + [P] * 4, I),
    "mpp3d_compute_vertex_connection_laplacian": ([P] * 6 + [I] * 3 + [P] * 6, I),
    "mpp3d_compute_corner_angles_embedded": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_face_normals": ([P] * 6 + [I] * 3 + [P] * 2, None),
    "mpp3d_compute_vertex_normals": ([P] * 6 + [I] * 3 + [P] * 3, None),
    "mpp3d_compute_vertex_tangent_basis": ([P] * 6 + [I] * 3 + [P] * 4, None),
    "mpp3d_cotan_laplacian_triplets": ([P, P, I, F, P, P, P], I),
    "mpp3d_face_areas": ([P, P, I, P], I),
    "mpp3d_vertex_areas": ([P, P, I, I, P, P], I),
    "mpp3d_build_general_mesh": ([P, I, I] + [P] * 9 + [P] * 5, None),
    "mpp3d_general_write_faces": ([P] * 9 + [P], None),
    "mpp3d_general_write_twins": ([P] * 9 + [P, P], None),
    "mpp3d_general_write_halfedge_edge_lengths": ([P] * 9 + [P, P], None),
    "mpp3d_general_duplicate_face": ([P] * 9 + [I], I),
    "mpp3d_general_invert_orientation": ([P] * 9 + [I], None),
    "mpp3d_general_separate_to_new_edge": ([P] * 9 + [I, I], I),
    "mpp3d_general_flip": ([P] * 9 + [I], I),
    "mpp3d_mollify_intrinsic": ([P] * 9 + [P, I, F], F),
    "mpp3d_build_intrinsic_tufted_cover": (
        [P] * 9 + [P, I, I, P, P, P, P, I, I, I], I,
    ),
    "mpp3d_flip_to_delaunay": ([P] * 9 + [P, I, P, P, I, F], I),
    "mpp3d_pc_neighbors": ([P, I, I, P, P, P], None),
    "mpp3d_pc_normals": ([P, P, I, I, P, P, P], None),
    "mpp3d_pc_tangent_coordinates": ([P, P, P, I, I, P], None),
    "mpp3d_pc_local_triangulation": (
        [P, P, I, I, I, P, P, P, P, P], I,
    ),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    found = os.environ.get("MOJO")
    if found:
        return found.split()
    from shutil import which

    exe = which("mojo")
    if exe:
        return [exe]
    pixi = which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "mojo"]
    raise BuildError("mojo not found; run `pixi run build`")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-potpourri3d.so` if stale."""
    sources = sorted(SRC.rglob("*.mojo"))
    if not force and LIB.exists():
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return str(LIB)
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", str(SRC), str(SRC / "capi.mojo"),
        "-o", str(LIB),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not LIB.exists():
        raise BuildError(((proc.stderr or "") + (proc.stdout or "")).strip()[:6000])
    return str(LIB)


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data
