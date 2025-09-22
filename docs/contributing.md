# Contributing to jaxeffort

We welcome contributions to jaxeffort! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/jaxeffort.git
   cd jaxeffort
   ```

3. Install in development mode with dependencies:
   ```bash
   poetry install --with dev
   ```

## Development Workflow

### Running Tests

Run the test suite using pytest:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=jaxeffort

# Run specific test file
poetry run pytest tests/test_emulator_loading.py

# Run tests in parallel
poetry run pytest -n auto
```

### Code Quality

Before submitting a pull request, ensure your code passes quality checks:

```bash
# Format code with Black
poetry run black jaxeffort/ tests/

# Sort imports with isort
poetry run isort jaxeffort/ tests/

# Lint with ruff
poetry run ruff check jaxeffort/ tests/

# Fix linting issues automatically
poetry run ruff check --fix jaxeffort/ tests/
```

### Building Documentation

To build and preview the documentation locally:

```bash
# Install documentation dependencies
pip install mkdocs-material mkdocstrings[python] mike

# Generate documentation plots
python generate_doc_plots.py

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## Pull Request Process

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request on GitHub

### PR Requirements

- All tests must pass
- Code must be formatted with Black
- New features should include tests
- Documentation should be updated if applicable
- Commit messages should be clear and descriptive

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes (Google style)
- Keep line length to 100 characters
- Use meaningful variable names

## Testing Guidelines

- Write tests for new features
- Maintain or improve code coverage
- Use pytest fixtures for shared test data
- Test edge cases and error conditions

Example test structure:

```python
import pytest
import jaxeffort

def test_multipole_emulator_loading():
    """Test that multipole emulators load correctly."""
    emulator = jaxeffort.load_multipole_emulator("path/to/emulator")
    assert emulator is not None
    assert hasattr(emulator, 'get_Pl')

def test_invalid_parameters():
    """Test handling of invalid parameters."""
    with pytest.raises(ValueError):
        jaxeffort.W0WaCDMCosmology(h=-0.5)  # Invalid Hubble parameter
```

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Python version and key package versions
- Any relevant error messages or tracebacks

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Provide a clear use case
- Describe the expected behavior
- Consider implementing it yourself and submitting a PR

## Community

- Follow the [Code of Conduct](https://github.com/CosmologicalEmulators/jaxeffort/blob/main/CODE_OF_CONDUCT.md)
- Be respectful and constructive in discussions
- Help others when you can

## Release Process

Releases are managed by maintainers and follow semantic versioning:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Contact the maintainers
- Check existing issues and discussions

Thank you for contributing to jaxeffort!