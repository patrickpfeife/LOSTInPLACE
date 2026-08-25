# PyTorch Geometric tutorial

A from-scratch, exercise-driven introduction to PyTorch Geometric (PyG), building up toward the model described in `../project.md` (Sections 3 and 5). Assumes general ML/DL knowledge but no prior GNN or PyG experience.

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

- `pyg_tutorial.ipynb` — the tutorial. Most code cells are deliberately left as `# TODO` stubs with hints and doc links, not finished solutions — see the notebook's intro cell for how it's meant to be used.
- `data/egms_l2b_ascending_sample.csv` — a ~1km × 1km, 4772-point cutout (near Munich, 208 acquisition dates 2020–2024) taken from `test_egms/egms_density_test/data/EGMS_L2b_Ascending.zip`, kept small enough to load and iterate on instantly. Regenerate or resize it by filtering the full zip on `latitude`/`longitude` the same way (see notebook Section 4).

## Notes

- No GPU is needed or used here — the graphs are tiny by design. `torch.cuda.is_available()` returning `False` on the login node is expected, not a problem.
- This is a learning sandbox, not the real pipeline. It intentionally simplifies things the real design in `project.md` doesn't (fixed neighbour sets instead of per-step neighbour masking, a single scalar instead of full coefficient vectors, a random split instead of a geographic one, no ensemble/calibration/decomposition) — the notebook's final section spells out exactly what's simplified and why.
