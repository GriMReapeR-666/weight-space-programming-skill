"""Environment inspector utilities.

Provides :func:`inspect_environment` which returns a dict indicating
whether ETK/RTK commands are available.
"""
import shutil


def inspect_environment():
    """Check for presence of `etk` and `rtk` executables in PATH.

    Returns a dictionary like ``{'etk': True, 'rtk': False}``.
    """
    return {
        "etk": shutil.which("etk") is not None,
        "rtk": shutil.which("rtk") is not None,
    }
