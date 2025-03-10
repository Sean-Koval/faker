# Examples

This directory contains example configuration files for generating different types of synthetic conversations.

## Usage

To generate a dataset using an example configuration:

```bash
# Using the Python module directly
poetry run python -m faker generate -c examples/config.yaml

# Using Docker
docker-compose up
```

## Configuration Structure

The configuration files are in YAML format and support the following sections:

- `name`: Name of the dataset
- `version`: Version of the dataset
- `description`: Description of the dataset
- `llm`: Configuration for the LLM provider
- `dataset`: Configuration for the generated dataset
- `conversation`: Configuration for the generated conversations
- `templates`: Prompt templates for generating conversations