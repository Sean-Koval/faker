# CLAUDE.md - Agent Instructions for faker

## Project Overview
`faker` is a modular synthetic chat data generation tool built with Python.

## Build & Test Commands
```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_specific.py::test_function

# Type checking
poetry run mypy .

# Lint code
poetry run black --check .
poetry run isort --check .

# Format code
poetry run black .
poetry run isort .

# Run in development mode
poetry run python -m faker
```

## Code Style Guidelines
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Imports**: Group imports by source (standard lib, external, internal)
- **Formatting**: Black (line length 88) and isort for code formatting
- **Types**: Use type hints and mypy for type checking
- **Error Handling**: Use specific exception types and proper context
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Modular Design**: Keep modules single-purpose and composable
- **No dependencies** on langchain or llamaindex for prompt chaining