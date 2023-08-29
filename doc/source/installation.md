# Installation

## Install from source

Install Miniconda: [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Clone the repository.

```bash
git clone https://github.com/kurtamohler/spacetimelib.git && cd spacetimelib
```

Create conda environment.

```bash
conda env create -f environment.yaml -n spacetime && conda activate spacetime
```

Install SpacetimeLib.

```bash
pip install -e .
```

## How to import SpacetimeLib

To access SpacetimeLib, import it into your Python code:

```python
>>> import spacetime as st
```

Shorten the imported name to `st` for better code readability, but you can just
use the unshortened name if you want.
