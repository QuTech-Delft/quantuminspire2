"""quantuminspire.

SDK for the Quantum Inspire platform.
"""

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    pass
