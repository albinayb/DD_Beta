# DiffractionDiver

A desktop GUI for diffraction analysis of electron microscopy images. The launcher opens four modules:

- **Sliding FFT / Array Analysis**: slide an FFT window across an image, then run PCA, ICA or NMF.
- **Diffractogram Peak Fit**: pick a reflection, fit it and its mirror peak, and map spacing, angle and intensity.
- **Radial Profile Analysis**: compute radial profiles for every patch and scan through spacing.
- **Carbon Tools**: orientation maps, segmentation and domain-size distributions for graphitic materials.

Input images can be `.npy`, `.png`, `.tif`/`.tiff` or `.jpg`.

## Install and run

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 and all dependencies for you) and either Git or the repository as a ZIP download.

### 1. Install uv

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal afterwards, then check it worked with `uv --version`.

### 2. Get the code

```bash
git clone https://github.com/albinayb/DD_Beta.git
cd DD_Beta
```

No Git? On the GitHub page choose **Code → Download ZIP**, unzip it, and open a terminal in the unzipped folder (the one containing `pyproject.toml`).

### 3. Install dependencies

```bash
uv sync
```

The first run downloads Python and the packages, which takes a few minutes. It creates a local `.venv` folder.

### 4. Run

```bash
uv run diffraction-diver
```

## Updating

```bash
git pull
uv sync
```

## Troubleshooting

- `uv: command not found`: reopen the terminal after installing uv.
- Run every command from the folder that contains `pyproject.toml`.
- If `uv sync` fails to download packages, check your network or proxy settings.
