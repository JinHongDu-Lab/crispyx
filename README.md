# crispyx

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/crispyx?label=pypi&color=orange)](https://pypi.org/project/crispyx)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/crispyx?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BRIGHTGREEN&left_text=downloads)](https://pepy.tech/project/crispyx)
[![Tests](https://github.com/jinhongdu-lab/crispyx/actions/workflows/tests.yml/badge.svg)](https://github.com/jinhongdu-lab/crispyx/actions/workflows/tests.yml)

## Motivation

Genome-wide CRISPR screens routinely produce datasets with hundreds of thousands of cells and tens of thousands of genes. Standard single-cell analysis toolkits (Scanpy, Pertpy) load the entire count matrix into memory, requiring large RAM allocations and often making routine workflows impractical on laptops or shared compute nodes.

**crispyx** solves this by streaming data directly from on-disk AnnData (`.h5ad`) files. Quality control, normalisation, pseudo-bulk aggregation, and differential expression all operate without materialising the full matrix.

## Features

- **Streaming QC & preprocessing** – Filter cells, perturbations, and genes; normalise and log-transform; CSC-aware streaming with `format_mismatch_policy`; all without loading the full matrix into memory
- **Pseudo-bulk aggregation** – Average log expression and pseudo-bulk count matrices for effect size estimation, with optional batch correction (`batch_column`) that combines within-batch effects using inverse-variance meta-analysis
- **Differential expression** – t-test, Wilcoxon rank-sum (including batch-stratified / van Elteren test via `batch_column`), and negative binomial GLM with apeGLM LFC shrinkage; multi-core support for NB-GLM and configurable worker memory limits
- **Dimension reduction** – Memory-efficient PCA and KNN graph construction on backed data
- **Scanpy-compatible API & plotting** – Familiar `cx.pp`, `cx.pb`, `cx.tl`, and `cx.pl` namespaces; Scanpy-style rank genes plots, volcano, MA, PCA, UMAP, QC summaries, and overlap heatmaps
- **Data preparation utilities** – Edit backed metadata without loading X; standardise gene names; normalise perturbation labels; auto-detect metadata columns
- **HPC-ready** – Resume/checkpoint for long-running jobs; configurable `memory_limit_gb`; Docker and Singularity support

## Quick Start

```python
import crispyx as cx

# Open dataset without loading into memory
adata = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")

# Quality control with adaptive thresholds
adata = cx.pp.qc_summary(
    adata,
    perturbation_column="perturbation",
    min_genes=5,
    min_cells_per_perturbation=5,
)

# Differential expression
adata = cx.tl.rank_genes_groups(
    adata,
    perturbation_column="perturbation",
    method="wilcoxon",  # or "t-test", "nb_glm"
)

# Access results
print(adata.uns["rank_genes_groups"])
de_results = adata.uns["rank_genes_groups"].load()
```

For the full workflow (normalisation, PCA, pseudo-bulk, NB-GLM, LFC shrinkage, plotting, data preparation utilities), see the [Usage Guide](docs/usage.rst) and the [tutorial notebook](docs/notebooks/crispyx_tutorial.ipynb).

## Performance

Benchmarked across 12 CRISPR screen datasets (21k–1.97M cells), crispyx consistently outperforms Scanpy, Pertpy/PyDESeq2, and edgeR in both speed and memory:

| Metric | crispyx vs Scanpy | crispyx vs Pertpy/PyDESeq2 |
|---|---|---|
| **t-test** | **2–11× faster** | — |
| **Wilcoxon** | **2–43× faster** | — |
| **NB-GLM** | — | **2× faster**, completes where Pertpy OOMs |
| **Peak memory** | **2–6× lower** | Runs within 64 GB where Pertpy exceeds 120 GB |
| **Accuracy** | Pearson *r* > 0.999 vs Scanpy | Pearson *r* > 0.97 vs PyDESeq2 |

crispyx succeeds on **all 12 datasets**, while Scanpy times out or OOMs on the largest screens and Pertpy/edgeR fail on most genome-wide datasets.

<p align="center">
  <img src="benchmarking/figures/benchmark_figure.png" width="800" alt="Benchmark results: crispyx vs reference methods">
</p>

See [benchmarking/](benchmarking/) for full results and reproduction scripts.

## Installation

```bash
pip install crispyx
```

For development (editable install with all extras):

```bash
git clone https://github.com/jinhongdu-lab/crispyx.git
cd crispyx
pip install -e ".[test,benchmark,docs]"
```

crispyx supports Python 3.10–3.12 and is compatible with recent releases of the
scientific stack, including `anndata >= 0.13` and `pandas >= 3.0` (where string
metadata is stored on disk using the nullable-string encoding).

## Benchmarking

```bash
cd benchmarking
./run_benchmark.sh config/Adamson.yaml       # single dataset
./run_benchmark.sh config/*.yaml             # all datasets
```

See [benchmarking/README.md](benchmarking/README.md) for configuration options and output structure.

## Testing

```bash
pytest
```

## Documentation

```bash
sphinx-build docs docs/_build
```

## Acknowledgements

crispyx builds on the foundational work of [Scanpy](https://scanpy.readthedocs.io/) (Wolf *et al.*, 2018), [Pertpy](https://pertpy.readthedocs.io/), [PyDESeq2](https://pydeseq2.readthedocs.io/) (Muzellec *et al.*, 2023), and [AnnData](https://anndata.readthedocs.io/) (Virshup *et al.*, 2024). We gratefully acknowledge these projects for establishing the single-cell analysis ecosystem in Python; crispyx extends their APIs and algorithmic designs to enable memory-efficient, streaming computation for large-scale CRISPR screen datasets.

## Contributing

Suggestions, bug reports, and contributions are welcome! Please open an [issue](https://github.com/jinhongdu-lab/crispyx/issues) or submit a pull request.
