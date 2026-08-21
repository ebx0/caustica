"""focus_study — a CPU-runnable HIFU focus characterization app.

Runs one steady-state CW scenario end to end with the current library
(M0-M6b + M8: numpy backend, k-space PSTD solvers, CSG geometry, planner)
and writes a self-contained result folder: figures, metrics.json, raw
fields (.npz) and an HTML/Markdown report.

Deliberately avoids everything the library does not have yet: no GPU
(M7), no HDF5/resume (M10), no Study harness (M11).
"""

__version__ = "0.1.0"
