# Installation

## Pip

```bash
pip install spacetimelib
```

## From source

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
