"""recallix was renamed to recallic. This package installs it and keeps `import recallix` working."""
import warnings

warnings.warn("recallix is now recallic: pip install recallic and import recallic", DeprecationWarning, stacklevel=2)

from recallic import *  # noqa: F401,F403
from recallic import Memory, MemoryClient, RecallicError  # noqa: F401
RecallixError = RecallicError  # the error class name before the rename
