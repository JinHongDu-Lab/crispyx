Installation
============

Requirements
------------

* **Python ≥ 3.10**
* A working C compiler for Numba JIT (usually ships with your OS)
* Approximately 4 GB of RAM for typical datasets; larger genome-wide screens
  benefit from 16–64 GB

Install from PyPI
-----------------

.. code-block:: bash

   pip install crispyx

Install from source (development)
----------------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/jinhongdu-lab/crispyx.git
   cd crispyx
   pip install -e .

Optional extras
---------------

crispyx offers optional dependency groups for different use cases:

.. code-block:: bash

   # Testing dependencies (pytest, statsmodels, pydeseq2)
   pip install -e ".[test]"

   # Benchmarking dependencies (pertpy, pyyaml, psutil)
   pip install -e ".[benchmark]"

   # Documentation dependencies (sphinx, nbsphinx, etc.)
   pip install -e ".[docs]"

   # Install everything
   pip install -e ".[test,benchmark,docs]"

Conda environment
-----------------

A ``env.yml`` file is provided for setting up a full Conda environment
including R packages for edgeR benchmarking:

.. code-block:: bash

   conda env create -f env.yml
   conda activate pert
   pip install -e .

This installs R 4.1.1 with Bioconductor edgeR and rpy2 for cross-language
benchmarking.

Verifying the installation
--------------------------

.. code-block:: python

   import crispyx as cx
   print(cx.__version__)

   # Quick check: open a dataset without loading into memory
   adata = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")
   print(adata)
