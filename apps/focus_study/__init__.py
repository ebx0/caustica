"""focus_study — a CPU-runnable HIFU focus characterization app.

Runs one steady-state CW scenario end to end with the current library
(numpy backend, k-space PSTD solvers, CSG geometry, planner)
and writes a self-contained result folder: figures, metrics.json, raw
fields (.npz) and an HTML/Markdown report.

Deliberately avoids everything the library does not have yet: no GPU
, no HDF5/resume, no Study harness.
"""

__version__ = "0.1.0"
