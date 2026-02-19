# pySSMF

A Python package to run Slave-Spin Mean-Field (SSMF) simulations in strongly correlated electron systems.

## Developers

This section explains how to prepare a local development environment for `pySSMF`.

First, clone and enter in the repository in your local machine:

```bash
git clone https://github.com/Yanene/pySSMF.git
cd pySSMF
```

### Prerequisites

- Python `3.12+`
- `git`
- One environment manager: `venv` (built into Python) or `conda`

### Option A: `venv` Environment

#### Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,docu]'
```

**Note**: The editable install (`-e`) reflects local code changes immediately.

**Note 2**: Developer tools are installed through the `dev` extra in `pyproject.toml`.

#### Windows (PowerShell)

Open the Command Prompt:

```bash
python3 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e '.[dev,docu]'
```

### Option B: Conda Environment (both Linux and Windows)

```bash
conda create -n pyssmf python=3.12 -y
conda activate pyssmf
pip install --upgrade pip
pip install -e '.[dev,docu]'
```

### Verify the Setup

Run these commands from the project root:

```bash
python --version
pytest tests/
```

### Install and Manage with `uv`

For faster setup, we recommend using [`uv`](https://docs.astral.sh/uv/).

#### Install `uv`

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Create and sync the project environment

```bash
uv venv
uv sync --extra dev
```

Activate if you want an interactive shell in the env:

Linux (bash):

```bash
source .venv/bin/activate
```

Windows (Command Prompt):

```bash
.\.venv\Scripts\activate
```

#### Run commands with `uv`

```bash
uv run python --version
uv run pytest tests
```

#### (Optional) Dependency management with `uv`

```bash
uv add <package>
uv add --dev <package>
uv add --docu <package>
uv lock 
uv sync 
```
