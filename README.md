# Faker

A modular synthetic chat data generation system built with Python.

## Overview

Faker is designed to quickly generate synthetic conversational/chat datasets in a modular fashion. Users can define parameters, configure settings using natural language, and generate datasets for diverse use cases.

## Features

- Configurable conversation templates
- Integration with Gemini (Google Vertex AI)
- Customizable conversation structures and personas
- Export in various formats
- Docker support for deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faker.git
cd faker

# Install with Poetry
poetry install

# Set up environment variables
cp .env.example .env
```

## Usage

```python
from faker import ChatGenerator

# Initialize the generator with a configuration
generator = ChatGenerator.from_config("config.yaml")

# Generate a dataset
dataset = generator.generate(num_conversations=10)

# Export to a file
dataset.export("conversations.jsonl")
```

## Development

This project uses Poetry for dependency management and packaging.

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .
```

## License

MIT
