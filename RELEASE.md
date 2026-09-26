# Release Checklist

- [x] GitHub username set to `01pouria`.
- [x] MIT License selected and added.
- [ ] Confirm package name `hmmpy` is available on PyPI.
- [ ] Run `pytest`.
- [ ] Run `python -m build`.
- [ ] Run `python -m twine check dist/*`.
- [ ] Install the wheel in a fresh virtual environment.
- [ ] Publish once to TestPyPI and install from TestPyPI.
- [ ] Create the GitHub repository.
- [ ] Configure PyPI Trusted Publishing for GitHub Actions environment `pypi`.
- [ ] Create a GitHub Release after checks pass.

Suggested local commands:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```
