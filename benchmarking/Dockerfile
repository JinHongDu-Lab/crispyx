# Dockerfile for CRISPYx Benchmarking
# Provides reproducible environment with Python, R, and all dependencies
# 
# Build: DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile .
# Run:   docker run --rm -v $(pwd):/workspace crispyx-benchmark --help
#
# For faster rebuilds when only changing crispyx code (not dependencies):
#   DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile .
# The source code layers are at the end, so dependency layers stay cached.

# Enable BuildKit cache mounts for faster rebuilds
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="CRISPYx Benchmark"
LABEL org.opencontainers.image.description="Reproducible benchmarking environment for CRISPR screen analysis"
LABEL org.opencontainers.image.source="https://github.com/jaydu1/Streamlining-CRISPR-Screen-Analysis"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale to avoid R encoding issues
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Add /workspace to PYTHONPATH so benchmarking module is importable
ENV PYTHONPATH=/workspace

# Install system dependencies for R and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # R and build tools
    r-base \
    r-base-dev \
    # Build dependencies for Python packages
    build-essential \
    gcc \
    g++ \
    gfortran \
    # HDF5 for h5py/anndata
    libhdf5-dev \
    pkg-config \
    # Other dependencies
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    # Git for pip installs from repos
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor and R packages
# Pin versions for reproducibility
# Note: R 4.2 from Debian bookworm requires Bioconductor 3.16
# Use BuildKit cache mount to cache R packages across builds
RUN --mount=type=cache,target=/root/.cache/R \
    --mount=type=cache,target=/tmp/downloaded_packages \
    R -e "install.packages('BiocManager', repos='https://cloud.r-project.org/')" \
    && R -e "BiocManager::install(version='3.16', ask=FALSE)" \
    && R -e "BiocManager::install(c('edgeR', 'DESeq2', 'apeglm'), ask=FALSE, update=FALSE)" \
    && R -e "install.packages('RhpcBLASctl', repos='https://cloud.r-project.org/')"

# Create workspace directory
WORKDIR /workspace

# ============================================================================
# DEPENDENCY INSTALLATION (cached layers - rarely change)
# ============================================================================
# These layers are intentionally placed BEFORE any COPY commands to maximize
# cache hit rate. Dependencies only reinstall when the RUN commands change.

# Install core dependencies first (for caching)
# Use BuildKit cache mount to cache pip downloads across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "numpy>=1.23" \
    "scipy>=1.10,<=1.12.0" \
    "pandas>=1.5" \
    "anndata>=0.9" \
    "h5py>=3.0" \
    "numba>=0.59" \
    "joblib>=1.0" \
    "scikit-learn>=1.0" \
    "scikit-misc>=0.1.4" \
    "scanpy>=1.9.2" \
    "seaborn>=0.12" \
    "matplotlib>=3.5" \
    "statsmodels>=0.14"

# Install DE/benchmark dependencies (including pertpy with ott-jax)
# This layer is slow but only rebuilds when dependency versions change
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "pydeseq2>=0.4" \
    "pertpy>=0.4" \
    rpy2 \
    "psutil>=5.9" \
    "pyyaml>=6.0" \
    "tqdm>=4.65" \
    filelock

# ============================================================================
# SOURCE CODE INSTALLATION (frequently changing layers - at the end)
# ============================================================================
# Copy source code LAST so that code changes don't invalidate dependency cache

# Copy pyproject.toml and source code
COPY pyproject.toml /workspace/
COPY src/ /workspace/src/

# Install crispyx in editable mode (fast - dependencies already installed)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e /workspace

# Copy benchmarking code
COPY benchmarking/ /workspace/benchmarking/

# Copy worker entrypoint and profiling module to /app directory
RUN mkdir -p /app \
    && cp /workspace/benchmarking/tools/docker_worker.py /app/docker_worker.py \
    && cp /workspace/benchmarking/tools/profiling.py /app/profiling.py

# Set working directory for benchmark execution
WORKDIR /workspace

# Default entrypoint runs the docker worker
ENTRYPOINT ["python", "/app/docker_worker.py"]

# Default command shows help
CMD ["--help"]
