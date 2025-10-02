# syntax=docker/dockerfile:1

# Allow pinning the base image digest at build time via --build-arg
ARG PY_BASE=python:3.11-slim

##############################
# Builder: build wheels only #
##############################
FROM ${PY_BASE} AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip && \
    pip wheel -r requirements.txt -c constraints.txt -w /wheels

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel .[service] -c constraints.txt -w /wheels

##############################
# Runtime (non-root)         #
##############################
FROM ${PY_BASE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd -m -u 10001 app
WORKDIR /app

# Copy wheels and install from wheels only (no network)
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels "ml-portfolio-churn[service]" && \
    rm -rf /wheels

# Copy runtime assets and set ownership
COPY --chown=app:app config.sample.json ./

# Ensure working dirs exist and owned by app
RUN mkdir -p /app/data /app/models && chown -R app:app /app

# Drop root
USER app

# Default command prints help; override in `docker run` with desired args
CMD ["churn", "--help"]

