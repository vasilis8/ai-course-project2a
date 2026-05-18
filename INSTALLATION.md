# Installation

## Prerequisites
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or Anaconda installed

## Setup

**1. Create and activate a conda environment with Python 3.10:**
```bash
conda create -n ai-project2a python=3.10 -y
conda activate ai-project2a
```

**2. Downgrade pip and setuptools** (required for gym==0.21.0 compatibility):
```bash
pip install "setuptools==65.5.0" "pip==21"
```

**3. Install ns-gym and all dependencies:**
```bash
pip install ns-gym --no-cache-dir
```

## Verify the Installation

Run the test script from inside the `AI-agents-in-ns-gym/` directory:
```bash
cd AI-agents-in-ns-gym
python3 test/test_ns_gymnasium.py
```

Expected output: a rendered gymnasium environment window opens and a list of random integers (actions) is printed to the terminal.

## Note on pyenv

If you have pyenv installed, `conda activate` may not fully override the Python shims. If you see the wrong Python path, use the conda env's Python directly:
```bash
/Users/<your-username>/miniconda3/envs/ai-project2a/bin/python
```
