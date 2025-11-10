# self-play

## Setup

### uv
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Download it using the standalone installer:
```
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or, from PyPi:
```
# With pip
pip install uv
```
Once uv is installed, create a virtual environment and activate it:
```
uv venv
# Using macOS/Linux
source .venv/bin/activate
```
Setup the dependencies: 
```
uv sync
```

### Data
We use [minari](https://minari.farama.org/)'s Kitchen environment to build our dataset. 