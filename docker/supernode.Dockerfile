FROM flwr/supernode:1.17.0-py3.10-ubuntu24.04

# Switch to root user to install build dependencies
USER root

# Install build-essential package for gcc (needed for scikit-learn)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER app

WORKDIR /app

# Copy the entire project
COPY --chown=app:app . .

# Install the package
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-supernode"]
