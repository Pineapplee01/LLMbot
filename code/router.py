"""Compatibility shim for the conformal risk routing package.

New code should import from `conformal_risk_routing.router`.
"""

from conformal_risk_routing.router import *  # noqa: F401,F403
from conformal_risk_routing.router import __all__  # noqa: F401
