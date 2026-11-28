"""recallix-adk was renamed to recallic-adk. This package installs it and keeps `import recallix_adk` working."""
import warnings

warnings.warn("recallix-adk is now recallic-adk: pip install recallic-adk and import recallic_adk", DeprecationWarning, stacklevel=2)

from recallic_adk import *  # noqa: F401,F403
from recallic_adk import RecallicMemoryService  # noqa: F401
RecallixMemoryService = RecallicMemoryService  # the class name before the rename
