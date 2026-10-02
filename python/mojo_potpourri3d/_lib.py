"""ctypes bridge to the fixed-ABI Mojo kernels in dist/libmojo-potpourri3d.so.

A Mojo function cannot keep state across the C ABI, so every solver here is three calls: a setup
call that builds the operators into caller-owned arenas and reports how big the factor arena has to
be, a factor call that Cholesky-factorizes, and then one compute call per query. The arena layouts
are fixed by the `*Layout` structs in the corresponding kernel module; the offset formulas below
mirror them.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "dist" / "libmojo-potpourri3d.so"

I = ctypes.c_int64
F = ctypes.c_double
P = ctypes.c_void_p

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list, object]] = {
    "pp3d_chol_analyze": ([I, I, P, P, P, P], I),
    "pp3d_chol_factorize": ([I, I, P, P, P, P, P, P, P, P], I),
    "pp3d_chol_solve": ([I, I, P, P, P, P, P, P, P], I),
    "pp3d_heat_layout": ([I, I, I, I, P], I),
    "pp3d_heat_setup": ([I, I, P, P, F, I, P, P, P], I),
    "pp3d_heat_factor": ([I, I, I, I, P, P, P, P, I, I], I),
    "pp3d_heat_compute_distance": ([I, I, I, I, P, P, P, P, P, P, P, I, P], I),
    "pp3d_cotan_laplacian": ([I, I, P, P, F, P, P, P], I),
    "pp3d_face_areas": ([I, I, P, P, P], I),
    "pp3d_vertex_areas": ([I, I, P, P, P], I),
    "pp3d_edges": ([I, I, P, P], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    found = os.environ.get("MOJO")
    if found:
        return [found, "build"]
    import shutil

    exe = shutil.which("mojo")
    if exe:
        return [exe, "build"]
    raise BuildError("mojo not found; run `pixi run build`")


_BUILD_SOURCES = (
    "ported.mojo", "ffi.mojo", "heat_method_distance.mojo", "intrinsic_geometry_interface.mojo",
    "local_triangulation.mojo", "mesh.mojo", "point_position_geometry.mojo", "simple_idt.mojo",
    "sparse.mojo", "sparse_matrix.mojo", "surface_mesh.mojo", "tufted_laplacian.mojo",
    "vector_util.mojo",
)


def build(force: bool = False) -> str:
    """Compile src/ported.mojo into dist/libmojo-potpourri3d.so if stale.

    Staleness is judged against every source the shared library is compiled from, not just the entry
    point. `mojo build` only emits from the file it is given, but the kernels it pulls in are the rest
    of src/*.mojo, so comparing against ported.mojo alone leaves a changed kernel unbuilt.
    """
    src = ROOT / "src"
    sources = [src / name for name in _BUILD_SOURCES]
    if not force and LIB.exists():
        built = LIB.stat().st_mtime
        if all(s.exists() and s.stat().st_mtime <= built for s in sources):
            return str(LIB)
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = mojo_command() + [
        "--emit",
        "shared-lib",
        "-I",
        str(src),
        str(sources[0]),
        "-o",
        str(LIB),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(proc.stdout + proc.stderr)
    return str(LIB)


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is not None:
        return _loaded
    build()
    dll = ctypes.CDLL(str(LIB))
    for name, (argtypes, restype) in _SIGNATURES.items():
        fn = getattr(dll, name)
        fn.argtypes = argtypes
        fn.restype = restype
    _loaded = dll
    return dll


def f64(a) -> np.ndarray:
    return _contiguous(a, np.float64)


def i64(a) -> np.ndarray:
    return _contiguous(a, np.int64)


def _contiguous(a, dtype) -> np.ndarray:
    """A C-contiguous array of `dtype` the kernels can index by element offset.

    `np.ascontiguousarray` already does the right thing, including copying a non-contiguous or
    differently-typed input. It is spelled out here because a cast of a strided array materialises a
    temporary: a caller that keeps only a view of its own array (V[::2], say) would otherwise have
    that temporary freed the moment the call returns, and the kernels would then write through a
    freed buffer. Every call site below holds the returned array in a local for the whole call, so
    the buffer stays alive.
    """
    return np.ascontiguousarray(a, dtype=dtype)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


# --- arena layouts, mirroring HeatLayout in src/heat_method_distance.moj0


def heat_layout(n_v: int, n_f: int, n_fc: int, n_ec: int) -> list[int]:
    """The arena layout, straight from the kernel, so the two never drift apart.

    Returns [w_size, m_size, w_hvif, w_hcw, w_corner_len, w_heat_ax, w_pois_ax, w_work,
    m_heat_ap, m_heat_ai, m_heat_perm, m_pois_ap, m_pois_ai, m_pois_perm, m_corner_v, m_first_he,
    m_corner_v_compute]: 17 int64 values, in the order heat_layout writes them.
    """
    out = np.zeros(17, dtype=np.int64)
    rc = lib().pp3d_heat_layout(n_v, n_f, n_fc, n_ec, out.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"heat layout failed with code {rc}")
    return [int(x) for x in out]


def edge_counts(n_v: int, faces: np.ndarray) -> tuple[int, int]:
    """Number of distinct edges and boundary edges of a triangulation."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return int(uniq.shape[0]), int((counts == 1).sum())
