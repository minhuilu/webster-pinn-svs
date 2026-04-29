# exp/A_static/phys_consts.py
"""
Physical constants used across Stage-A code.
All modules should import Lx and c from here.
Units:
  - x: meters
  - t: seconds
  - c: meters/second
"""
# Optional: prefer L from the speaker profile if available
try:
    from .common.singer import get_profile
    _PROFILE = get_profile()
    Lx = float(_PROFILE.get("L", 0.155))   # effective vocal-tract length in meters
except Exception:
    Lx = 0.155

c = 343.0  # sound speed in m/s, common value at 20 C

__all__ = ["Lx", "c"]
