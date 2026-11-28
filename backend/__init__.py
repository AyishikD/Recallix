"""Recallic backend."""
import os as _os

# The project was called Recallix before; settings named RECALLIX_* still work (RECALLIC_* wins when both are set).
for _k, _v in list(_os.environ.items()):
    if _k.startswith("RECALLIX_") and ("RECALLIC_" + _k[9:]) not in _os.environ:
        _os.environ["RECALLIC_" + _k[9:]] = _v
