"""recallix-haystack was renamed to recallic-haystack. This package installs it and keeps `import recallix_haystack` working."""
import warnings

warnings.warn("recallix-haystack is now recallic-haystack: pip install recallic-haystack and import recallic_haystack", DeprecationWarning, stacklevel=2)

from recallic_haystack import *  # noqa: F401,F403
from recallic_haystack import RecallicMemoryStore  # noqa: F401
RecallixMemoryStore = RecallicMemoryStore  # the class name before the rename
