"""Applications built ON TOP of the caustica library.

Everything here is a consumer of the public API — apps never reach into
``caustica`` internals. They are the honest end-to-end check that the
library is usable as shipped, and they are excluded from the library's
own test/lint gates by design (see apps/README.md).
"""
