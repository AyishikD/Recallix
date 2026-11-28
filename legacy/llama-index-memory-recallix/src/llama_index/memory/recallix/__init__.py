"""llama-index-memory-recallix was renamed to llama-index-memory-recallic. This package installs it and keeps `import llama_index.memory.recallix` working."""
import warnings

warnings.warn("llama-index-memory-recallix is now llama-index-memory-recallic: pip install llama-index-memory-recallic and import llama_index.memory.recallic", DeprecationWarning, stacklevel=2)

from llama_index.memory.recallic import *  # noqa: F401,F403
from llama_index.memory.recallic import RecallicMemoryBlock  # noqa: F401
RecallixMemoryBlock = RecallicMemoryBlock  # the class name before the rename
