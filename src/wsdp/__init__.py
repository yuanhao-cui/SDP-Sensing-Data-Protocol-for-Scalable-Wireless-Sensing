"""Compatibility facade for MVP reader imports.

The MVP package lives under :mod:`sdp_mvp`; this module keeps the original
``wsdp`` reader import path usable for code migrated from ``main``.
"""

from sdp_mvp import *  # noqa: F403
from sdp_mvp import __all__ as __all__
