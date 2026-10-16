## Build and upload a new distribution to PyPI

Update version numbers in `pyproject.toml` and `setup.py`.

Build the distribution.

```bash
rm -rf build dist *.egg-info
python -m pip install --upgrade build
python -m build
```

Upload to PyPI. This will require access to an API token for the spacetimelib
project on PyPI, which you can either paste in when prompted or you can set up
a `~/.pypirc` file ahead of time.

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```
