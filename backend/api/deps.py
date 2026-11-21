import threading

_lock = threading.Lock()
_memory_manager = None
_scheduler = None


def get_memory_manager():
    """Process-wide MemoryManager (created on first use)."""
    global _memory_manager
    if _memory_manager is None:
        with _lock:
            if _memory_manager is None:
                from backend.brain.memory.manager import MemoryManager
                _memory_manager = MemoryManager()
    return _memory_manager


def get_scheduler():
    global _scheduler
    if _scheduler is None:
        with _lock:
            if _scheduler is None:
                from backend.brain.workers.scheduler import Scheduler
                _scheduler = Scheduler(get_memory_manager())
    return _scheduler


def reset():
    """Drop singletons (tests)."""
    global _memory_manager, _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.stop()
        _memory_manager = None
        _scheduler = None
