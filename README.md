# Quantum Many-Body Distance Learning

Supporting data and code for the paper  
**“Distance learning from projective measurements as an information-geometric probe of many-body physics.”**

At the moment this repository is under active development.

# What this repository contains
- Data and notebooks to reproduce paper figures (see `paper_figures/`).
- The `born2disc` Python package as a local uv workspace member (see `packages/born2disc/`).
- A script `download_data.sh` downloading the required data from [Zenodo](https://zenodo.org/records/18892338) (see below).

# Installation (uv)

```bash
# install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# enter the repo
cd quantum_many_body_distance_learning

# optional: install a specific Python version via uv
# uv python install 3.13

# create and activate environment
# if you installed Python with uv, you can pin it:
# uv venv --python 3.13
uv venv
source .venv/bin/activate

# install project dependencies and workspace members (including born2disc)
uv sync
```

# Installation (pip)

If you do not use `uv`, you can install the everything with `pip`:

```bash
# enter the repo
cd quantum_many_body_distance_learning

# create and activate a virtual environment with Python 3.13+
python3.13 -m venv .venv
source .venv/bin/activate

# upgrade packaging tools, then install the local package and notebook deps
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

This installs:
- the local `born2disc` package in editable mode from `packages/born2disc/`,
- the notebook and plotting dependencies used in this repository,
- the Git-based `nestconf` dependency required by `born2disc`.

# Verify born2disc is available

```bash
uv run python -c "import born2disc; print(born2disc.__file__)"
uv run python -c "from born2disc.pipelines.distance_learning_pipeline import DistanceLearningPipeline"
```

For a `pip` environment, use plain `python` instead:

```bash
python -c "import born2disc; print(born2disc.__file__)"
python -c "from born2disc.pipelines.distance_learning_pipeline import DistanceLearningPipeline"
```

# Download data

The repository expects processed figure data under `data/figures/` and raw TFIM
snapshots under `data/tfim/raw_snapshots/`.

Run:

```bash
./scripts/download_data.sh
```

To replace existing local data, add `--force`.

# What to run

After downloading the Zenodo data, the primary runnable example is
`toy_examples/tfim.ipynb`. That notebook is expected to work directly with the
downloaded toy-example data and prepares the local TFIM snapshot infrastructure
used by the archived Figure 3 runner.

After the toy example has been executed, you can run
`data_zenodo/figures/figure_3/main_distance_matrices/run_figure_3_main_distance_matrices.py`.
That archived script is the only `data_zenodo` runner in this repository with a
filled default snapshot path. It writes its outputs and copied script to
`toy_examples/attempt/`.

The other archived `data_zenodo/figures/**/run_*.py` scripts are kept primarily
as references for the published hyperparameters. By default they intentionally do
not ship with filled snapshot paths and will raise an error unless you set their
corresponding `BORN2DISC_*_SNAPSHOTS_ROOT` environment variable to a compatible
local dataset.
