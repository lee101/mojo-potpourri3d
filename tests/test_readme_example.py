"""Executes the usage block in README.md, so the documented example cannot rot.

The block is indented rather than fenced, matching the rest of the file. The
`## Usage` section holds two indented regions: a short one showing the
`read_mesh` shape (which needs a mesh file this repo does not carry), and the
self-contained icosahedron example, which is what we run.
"""

import pathlib

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _usage_runs() -> list[str]:
    """Indented lines of the `## Usage` section, split into runs at blank lines."""
    section = README.read_text().split("## Usage", 1)[1].split("\n## ", 1)[0]
    runs: list[str] = []
    current: list[str] = []
    for line in section.splitlines():
        if line.startswith("    "):
            current.append(line[4:])
        elif line.strip():
            if current:
                runs.append("\n".join(current))
                current = []
    if current:
        runs.append("\n".join(current))
    return runs


def _self_contained_example() -> str:
    """The icosahedron example: the last `import numpy as np` run to the end."""
    runs = _usage_runs()
    starts = [i for i, r in enumerate(runs) if r.startswith("import numpy as np")]
    assert starts, "no import block under '## Usage'"
    return "\n".join(runs[starts[-1]:])


def test_readme_has_a_runnable_usage_block():
    src = _self_contained_example()
    assert "mojo_potpourri3d" in src
    for call in (
        "pp3d.compute_distance",
        "compute_distance_multisource",
        "pp3d.cotan_laplacian",
        "get_tangent_frames",
        "get_connection_laplacian",
        "transport_tangent_vector",
        "extend_scalar",
    ):
        assert call in src, call


def test_readme_usage_block_runs():
    src = _self_contained_example()
    ns: dict = {}
    exec(compile(src, "README.md:usage", "exec"), ns)

    import numpy as np

    assert ns["d"][0] == 0.0
    assert ns["dm"].shape == (12,)
    assert ns["L"].shape == (12, 12)
    # a closed mesh has the constants in the cotan Laplacian's nullspace
    assert abs(ns["L"] @ np.ones(12)).max() < 1e-12
    # the antipode is farthest, and the heat method underestimates the geodesic
    assert 2.6 < ns["d"].max() < 2.8
    # transported directions stay unit length
    assert abs(np.linalg.norm(ns["u"], axis=1) - 1.0).max() < 1e-12
    for arr in (ns["bX"], ns["bY"], ns["n"]):
        assert arr.shape == (12, 3)
