# SpacetimeLib

SpacetimeLib is a special relativity physics library for Python.

SpacetimeLib performs mathematical operations on events, velocities, and
worldlines in N+1 Minkowski spacetime.

You can calculate Lorentz transformations, time dilation, length contraction,
and more.

## Documentation and getting started

The documentation for the latest `main` branch is found here:
[https://kurtamohler.github.io/spacetimelib/](https://kurtamohler.github.io/spacetimelib/)

The [Start Here - Twin Paradox
tutorial](https://kurtamohler.github.io/spacetimelib/notebooks/Twin%20Paradox.html)
page is a good starting point to see what SpacetimeLib can do.

## Installation

### Pip

```bash
pip install spacetimelib
```

### From source

Install Miniconda:
[instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Clone SpacetimeLib the repository.

```bash
https://github.com/kurtamohler/spacetimelib.git && cd spacetimelib
```

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n spacetimelib && conda activate spacetimelib
```

Then install SpacetimeLib.

```bash
pip install -e .
```

## How to import SpacetimeLib

To access SpacetimeLib, import it into your Python code:

```python
>>> import spacetimelib as st
```

Shorten the imported name to `st` for better code readability, but you can just
use the unshortened name if you want.
