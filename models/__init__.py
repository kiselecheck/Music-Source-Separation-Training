"""Source-checkout compatibility alias for :mod:`msst.models`."""

from warnings import warn

from msst import models as _models

warn(
    "Import models from 'msst.models'; the top-level 'models' package is deprecated.",
    DeprecationWarning,
    stacklevel=2,
)

__path__ = _models.__path__
