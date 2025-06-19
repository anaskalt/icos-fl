"""Enhanced singleton metaclass with parameterized instance support.

This module provides a thread-safe singleton implementation that creates
separate instances for different parameter combinations while maintaining
singleton behavior for identical parameters.
"""

# ruff: noqa: ANN401

import threading
from typing import ClassVar, Dict, Tuple, Type


class Singleton(type):
    """Thread-safe singleton metaclass supporting parameterized instances.

    Creates separate singleton instances for different initialization parameters
    while ensuring identical parameters always return the same instance.

    Key Features:
    - Parameter-aware instance creation
    - Thread-safe implementation with proper locking
    - Minimal overhead and clean implementation

    Usage:
        class DataClayConnection(metaclass=Singleton):
            def __init__(self, host: str = "localhost", port: int = 6867):
                self.host = host
                self.port = port

        # Same parameters return same instance
        conn1 = DataClayConnection("server1", 6867)
        conn2 = DataClayConnection("server1", 6867)
        assert conn1 is conn2  # True

        # Different parameters create different instances
        conn3 = DataClayConnection("server2", 6867)
        assert conn1 is not conn3  # True
    """

    # Class-level storage: key = (class, args_tuple, kwargs_tuple), value = instance
    _instances: ClassVar[
        Dict[Tuple[Type, Tuple[object, ...], Tuple[Tuple[str, object], ...]], object]
    ] = {}

    # Per-class locks for thread-safe instance creation
    _locks: ClassVar[Dict[Type, threading.Lock]] = {}

    def __call__(cls, *args: object, **kwargs: object) -> object:
        """Create or return singleton instance based on class and parameters.

        This method handles instance creation with parameter-based differentiation,
        ensuring thread-safe singleton behavior for each unique parameter combination.

        Args:
            *args: Positional arguments for class initialization
            **kwargs: Keyword arguments for class initialization

        Returns:
            Singleton instance for the given class and parameter combination
        """
        # Create unique key from class type and parameters
        # Sort kwargs for consistent key generation regardless of order
        key = (cls, args, tuple(sorted(kwargs.items())))

        # Fast path: return existing instance without acquiring lock
        if key in cls._instances:  # type: ignore[attr-defined]
            return cls._instances[key]  # type: ignore[attr-defined]

        # Ensure per-class lock exists for thread-safe instance creation
        if cls not in cls._locks:  # type: ignore[attr-defined]
            # Thread-safe lock creation using double-checked locking pattern
            with threading.Lock():
                if cls not in cls._locks:  # type: ignore[attr-defined]
                    cls._locks[cls] = threading.Lock()  # type: ignore[attr-defined]

        # Thread-safe instance creation with double-checked locking
        with cls._locks[cls]:  # type: ignore[attr-defined]
            # Re-check after acquiring lock to prevent race conditions
            if key in cls._instances:  # type: ignore[attr-defined]
                return cls._instances[key]  # type: ignore[attr-defined]

            # Create new instance using parent metaclass __call__ method
            instance = super().__call__(*args, **kwargs)  # type: ignore[misc]
            cls._instances[key] = instance  # type: ignore[attr-defined]

        return cls._instances[key]  # type: ignore[attr-defined]
