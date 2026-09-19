# Tutorials

Exercise-driven notebooks for this thesis's codebase. Same convention throughout: most code cells are `# TODO` stubs with hints and doc links, not finished solutions — each notebook's intro cell explains exactly how it's meant to be used.

- **`pyg_tutorial.ipynb`** — from-scratch introduction to PyTorch Geometric (PyG), building up toward the model described in `../project.md` (Sections 3 and 5). Assumes general ML/DL knowledge but no prior GNN or PyG experience.
- **`oop_tutorial.ipynb`** — object-oriented Python past the basics (inheritance/`super()`, dunder methods, `@property`/`@classmethod`/`@staticmethod`, `abc.ABC`, `@dataclass`, `__call__`), using the same EGMS sample data and building toward `project.md` Section 8.2's baseline-comparison design. Assumes you already know classes/`self`/`__init__` — it picks up from there, and assumes the Python level `pyg_tutorial.ipynb` requires.

## Setup

Environment is managed with `uv`:

```bash
module load uv          # or however uv is available on your system
uv sync                 # installs the exact locked environment (torch, torch-geometric, jupyter, ...)
```

Run the notebook either via JupyterLab:

```bash
uv run jupyter lab
```

or in VS Code / another IDE: select the **"LOSTInPLACE Tutorial"** kernel (registered under `~/.local/share/jupyter/kernels/lostinplace-tutorial`). If it's not showing up, re-register it with:

```bash
uv run python -m ipykernel install --user --name lostinplace-tutorial --display-name "LOSTInPLACE Tutorial"
```

## Contents

- `data/egms_l2b_ascending_sample.csv` — a ~1km × 1km, 4772-point cutout (near Munich, 208 acquisition dates 2020–2024) taken from `test_egms/egms_density_test/data/EGMS_L2b_Ascending.zip`, kept small enough to load and iterate on instantly. Shared by both notebooks. Regenerate or resize it by filtering the full zip on `latitude`/`longitude` the same way (see `pyg_tutorial.ipynb` Section 4).
- `empty_scripts/` — pristine, unfilled-in mirrors of both notebooks (same stubs, none of your own in-progress answers), to reset from if you want a clean run at an exercise.

## Notes

- No GPU is needed or used here — the graphs are tiny by design. `torch.cuda.is_available()` returning `False` on the login node is expected, not a problem.
- These are learning sandboxes, not the real pipeline. `pyg_tutorial.ipynb` intentionally simplifies things the real design in `project.md` doesn't (fixed neighbour sets instead of per-step neighbour masking, a single scalar instead of full coefficient vectors, a random split instead of a geographic one, no ensemble/calibration/decomposition) — its final section spells out exactly what's simplified and why. `oop_tutorial.ipynb`'s `Interpolator`/`NeighborFinder`/`RMSEScorer` classes are likewise a small, from-scratch stand-in for what a real baseline-comparison module would look like, not code meant to be copied into the actual pipeline as-is.
