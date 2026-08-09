API Reference
=============

crispyx exposes a Scanpy-style API through four namespace singletons:

* ``cx.pp`` — Preprocessing (QC, normalisation, PCA, neighbours, format conversion)
* ``cx.pb`` — Pseudo-bulk aggregation
* ``cx.tl`` — Tools (differential expression, LFC shrinkage, overlap analysis)
* ``cx.pl`` — Plotting

Most functions also accept file paths or backed AnnData objects directly.

Namespace API
-------------

Preprocessing (``cx.pp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PreprocessingNamespace
   :members:
   :undoc-members:

Pseudo-bulk (``cx.pb``)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PseudobulkNamespace
   :members:
   :undoc-members:

Tools (``cx.tl``)
~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._ToolsNamespace
   :members:
   :undoc-members:

Plotting (``cx.pl``)
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PlottingNamespace
   :members:
   :undoc-members:


Module Reference
----------------

The sections below document the underlying modules used by the namespace
API.  These are useful for advanced usage or for understanding parameter
details.

Data loading and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.data
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Disk-usage estimation
~~~~~~~~~~~~~~~~~~~~~

crispyx automatically warns -- without blocking the call -- when a
streaming write's disk footprint looks large relative to free space. To
check usage *before* committing to a run, call ``estimate_disk_usage`` with
the function you intend to run, its input file, and the same keyword
arguments you plan to pass. Also available as ``cx.tl.estimate_disk_usage``
(see the Tools namespace above). See :ref:`disk-space` in the usage guide
for the full explanation and a worked example.

.. autofunction:: crispyx.estimate_disk_usage

Quality control
~~~~~~~~~~~~~~~

.. automodule:: crispyx.qc
   :members:
   :undoc-members:
   :show-inheritance:

Pseudo-bulk aggregation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.pseudobulk
   :members:
   :undoc-members:
   :show-inheritance:

Generic batch statistics
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.batch
   :members:
   :undoc-members:
   :show-inheritance:

Differential expression
~~~~~~~~~~~~~~~~~~~~~~~

All three DE functions accept a ``force: bool = False`` parameter (v0.0.3+).
When the expected output ``.h5ad`` file already exists on disk, the function
reloads and returns the saved result instead of rerunning.  Pass ``force=True``
to overwrite.  See :ref:`auto-reload` in the usage guide.

.. automodule:: crispyx.de
   :members:
   :undoc-members:
   :show-inheritance:

Negative binomial GLM
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.glm
   :members:
   :undoc-members:
   :show-inheritance:

Dimension reduction
~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.dimred
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

.. automodule:: crispyx.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Profiling
~~~~~~~~~

.. automodule:: crispyx.profiling
   :members:
   :undoc-members:
   :show-inheritance:
