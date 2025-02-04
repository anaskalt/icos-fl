"""Singleton metaclass implementation."""
from typing import Dict, Any

class Singleton(type):
    """Metaclass for ensuring only one instance of a class exists.
    
    Maintains a dictionary of instances and returns existing instance if available.
    """
    
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        """Return existing instance or create new one."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]