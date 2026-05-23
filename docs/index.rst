crispyx
=======

.. image:: https://img.shields.io/pypi/v/crispyx?label=pypi&color=orange
   :target: https://pypi.org/project/crispyx
   :alt: PyPI

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Genome-wide CRISPR screens routinely produce datasets with hundreds of thousands
of cells and tens of thousands of genes. Standard single-cell toolkits load the
entire count matrix into memory, which can require 30–100+ GB of RAM. **crispyx**
streams data directly from on-disk AnnData ``.h5ad`` files so that quality
control, normalisation, pseudo-bulk aggregation, and differential expression all
run without materialising the full matrix — even the largest screens can be
processed with modest resources.

The API mirrors Scanpy (``cx.pp``, ``cx.pb``, ``cx.tl``, ``cx.pl``) so existing
workflows can migrate with minimal changes. See the :doc:`tutorial <notebooks/crispyx_tutorial>`
for an end-to-end walkthrough.

Key features
------------

- **Streaming QC & preprocessing** — filter and normalise without loading
  the full matrix
- **Pseudo-bulk aggregation** — average log expression and pseudo-bulk
  count matrices
- **Differential expression** — t-test, Wilcoxon, NB-GLM with apeGLM LFC
  shrinkage
- **Dimension reduction** — memory-efficient PCA and KNN on backed data
- **Scanpy-compatible API** — familiar namespaces and plotting helpers
- **HPC-ready** — resume/checkpoint, configurable memory limits, Docker
  and Singularity

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage
   notebooks/crispyx_tutorial

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   benchmarking
   comparison
   faq

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
