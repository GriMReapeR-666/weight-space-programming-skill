import importlib as _importlib

# Expose the import_module function for test monkeypatching
import_module = _importlib.import_module
