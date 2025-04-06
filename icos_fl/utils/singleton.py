"""Import a Singleton class definition.

Make it a separate module so that it can be used from any class
that wants to operate as a singleton class.
"""

from typing import ClassVar, Dict, Type


class Singleton(type):
    """A class to pass as metaclass to a class definition in order to make it singleton.

    Usage:
        class MySingletonClass(MyParentClassIfAny, metaclass=Singleton):
            ...
            ...
    """

    _instances: ClassVar[Dict[Type, object]] = {}

    def __call__(cls, *args: object, **kwargs: object) -> object:
        """Create or return the singleton instance of the class."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
