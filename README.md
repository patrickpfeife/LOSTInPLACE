<div align="center">

# LOSTInPLACE

**L**ine **O**f **S**ight **T**o **IN**SAR-derived continuous Dis**PLACE**ment

*Query-anywhere ground displacement from Sentinel-1 InSAR, learned with a graph neural network*

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/env-uv-DE5FE9)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)

Master's thesis at the **German Aerospace Center (DLR)** and **Trier University**

</div>

---

## About

Given any coordinate in Europe, LOSTInPLACE predicts **east-west and vertical ground displacement time series with calibrated uncertainty**, not only at the sparse points where InSAR happens to measure.

- **Interpolate:** a graph attention network learns spatial interpolation of Sentinel-1 line-of-sight displacement ([EGMS](https://egms.land.copernicus.eu/) L2b), trained self-supervised by masking target points.
- **Decompose:** ascending and descending line-of-sight are converted to east-west and vertical with the classical geometric inversion.
- **Validate:** against GNSS time series (Nevada Geodetic Laboratory) and a regression-kriging baseline, with uncertainty from a heteroscedastic head, a deep ensemble and conformal calibration.

Full design and rationale: [`project.md`](project.md). Full project name: *LOSTInPLACE - Line Of Sight To INSAR-derived continuous DisPLACEment*.

## Reproducing the pipeline

1. **Environment:** Python 3.12 with [uv](https://docs.astral.sh/uv/) (never plain `pip`).
   ```bash
   cd scripts/python && uv sync
   ```
2. **EGMS access:** an EGMS API service key saved as `assets/token.jwt` (git-ignored, never commit it).
3. **Storage:** set `PROJECT_DIR` in `scripts/python/01_download_data.py` to a location with tens of GB free.
4. **Download:** `uv run 01_download_data.py`, or on SLURM via `scripts/slurm/01_download_data.slurm` (submit from `scripts/slurm/`, adjust partition and account to your cluster). Needs outbound HTTPS.
5. **Training:** PyTorch and PyG are installed with CUDA support; training is meant for a GPU node.

Currently only the download stage is implemented; later stages follow the plan in `project.md`.

## Repository map

| Path | Contents |
|---|---|
| `project.md` | Design document and thesis plan |
| `scripts/` | Pipeline code (`python/`) and SLURM jobs (`slurm/`) |
| `assets/` | Study regions and EGMS coverage boundaries |
| `tutorial/` | Learning notebooks: PyTorch Geometric, OOP |
| `presentation/` | Figure scripts for talks and slides |
| `test/` | Side experiments, e.g. EGMS vs. GNSS comparison |

License: see [`LICENSE`](LICENSE).
