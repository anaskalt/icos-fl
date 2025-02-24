"""Singleton metaclass implementation.

Provides a metaclass that ensures only one instance
of a class exists per unique identifier.
"""
from typing import Dict, Any, Tuple

class Singleton(type):
   """Metaclass for implementing the singleton pattern.
   
   Maintains a dictionary of instances keyed by (class, identifier) tuples
   to allow multiple named instances of the same class while ensuring
   only one instance exists per unique identifier.

   Attributes:
       _instances: Dictionary storing singleton instances
   """
   
   _instances: Dict[Tuple[type, Any], Any] = {}

   def __call__(cls, *args, **kwargs) -> Any:
       """Create new instance or return existing one.
       
       Uses the first argument as an identifier to allow multiple named
       instances.

       Args:
           *args: Variable length argument list
           **kwargs: Arbitrary keyword arguments

       Returns:
           Instance of the class, either existing or newly created
       """
       key = (cls, args[0] if args else None) 
       
       if key not in cls._instances:
           cls._instances[key] = super().__call__(*args, **kwargs)
           
       return cls._instances[key]