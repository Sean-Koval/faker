FROM python:3.10-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==1.7.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment in Docker
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Copy the project code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
CMD ["python", "-m", "faker"]