"""Source-checkout compatibility alias for :mod:`msst.utils`."""

from warnings import warn

from msst import utils as _utils

warn(
    "Import utilities from 'msst.utils'; the top-level 'utils' package is deprecated.",
    DeprecationWarning,
    stacklevel=2,
)

__path__ = _utils.__path__
