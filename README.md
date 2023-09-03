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

## Install

Install Miniconda:
[instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n spacetime && conda activate spacetime
```

Then install SpacetimeLib.

```bash
pip install -e .
```
