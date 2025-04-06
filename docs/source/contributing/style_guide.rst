===========
Style Guide
===========

This page describes the coding style and conventions used in ICOS-FL.

Code Formatting
---------------

ICOS-FL uses the following tools for code formatting:

* **Black**: Code formatter
* **isort**: Import sorter
* **Ruff**: Linter
* **mypy**: Type checker

These tools are configured in ``pyproject.toml`` and enforced via pre-commit hooks.

Black Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.black]
   line-length = 99
   target-version = ["py310", "py311", "py312"]

isort Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.isort]
   profile = "black"
   multi_line_output = 3
   include_trailing_comma = true
   force_grid_wrap = 0
   use_parentheses = true
   line_length = 99
   known_first_party = ["icos_fl"]

Ruff Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.ruff]
   target-version = "py310"
   output-format = "full"
   line-length = 99
   fix = true

   [tool.ruff.lint]
   select = [
     "E", "F", "W",  # flake8
     "C",            # mccabe
     "I",            # isort
     "N",            # pep8-naming
     "D",            # flake8-docstrings
     "ANN",          # flake8-annotations
     # ...
   ]

mypy Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.mypy]
   warn_return_any = true
   warn_unused_configs = true
   # ...

Naming Conventions
------------------

Follow these naming conventions:

Variables and Functions
~~~~~~~~~~~~~~~~~~~~~~~

* Use snake_case for variable and function names
* Use descriptive names that indicate purpose
* Avoid single-letter names except for loop indices

.. code-block:: python

   # Good
   time_series_data = fetch_data()
   normalized_data = normalize_data(time_series_data)

   # Bad
   ts = fetch_data()
   nd = normalize_data(ts)

Classes
~~~~~~~

* Use PascalCase for class names
* Use noun phrases that describe what the class represents

.. code-block:: python

   # Good
   class TimeSeriesDataset:
       pass

   class LSTMModel:
       pass

   # Bad
   class process_data:
       pass

   class data:
       pass

Constants
~~~~~~~~~

* Use UPPER_CASE for constants
* Constants should be defined at the module level

.. code-block:: python

   # Good
   MAX_SEQUENCE_LENGTH = 100
   DEFAULT_BATCH_SIZE = 64

   # Bad
   maxSequenceLength = 100
   default_batch_size = 64

Modules and Packages
~~~~~~~~~~~~~~~~~~~~

* Use lowercase for module and package names
* Use short, descriptive names
* Avoid underscores in module names

.. code-block:: python

   # Good
   from icos_fl.utils.processor import Processor

   # Bad
   from icos_fl.Utils.data_processor import Processor

Type Variables
~~~~~~~~~~~~~~

* Use PascalCase for type variables
* Use single-letter names for simple type variables
* Use descriptive names for complex type variables

.. code-block:: python

   # Good
   T = TypeVar('T')
   TensorData = TypeVar('TensorData', bound=torch.Tensor)

   # Bad
   t = TypeVar('t')
   tensor_data = TypeVar('tensor_data', bound=torch.Tensor)

Docstring Style
---------------

ICOS-FL uses Google-style docstrings:

.. code-block:: python

   def fetch_data(timeout: int = 60) -> pd.DataFrame:
       """Fetch time series data from DataClay.

       Args:
           timeout: Maximum time to wait for data in seconds

       Returns:
           DataFrame containing the processed time series data

       Raises:
           TimeoutError: If no data is available within the timeout period
           DataClayException: If there is an error connecting to DataClay

       Example:
           >>> fetcher = Fetcher()
           >>> df = fetcher.fetch_data(timeout=30)
           >>> print(df.shape)
           (300, 4)
       """

Class Docstrings
~~~~~~~~~~~~~~~~

.. code-block:: python

   class TimeSeriesDataset(Dataset):
       """Dataset for time series prediction with sliding window approach.

       Creates sequences of consecutive time steps as inputs and
       uses the next value as the prediction target.

       Attributes:
           data: The time series data
           time_step: Number of time steps in each sequence
           metric: The target metric column
           device: PyTorch device to place tensors on
       """

Module Docstrings
~~~~~~~~~~~~~~~~~

.. code-block:: python

   """Utilities for processing time series data.

   This module provides classes and functions for processing time series data
   for LSTM model training, including normalization, sequence creation, and
   DataLoader generation.
   """

File Layout
-----------

Follow this order for file contents:

1. Module docstring
2. Imports (grouped as described in Imports section)
3. Constants
4. Global variables
5. Classes
6. Functions
7. Main execution block (if applicable)

Import Conventions
------------------

Organize imports in the following groups, separated by a blank line:

1. Standard library imports
2. Third-party imports
3. Local application imports

Within each group, sort imports alphabetically.

.. code-block:: python

   # Standard library imports
   import os
   import sys
   import time
   from typing import Dict, List, Optional, Tuple

   # Third-party imports
   import numpy as np
   import pandas as pd
   import torch
   import torch.nn as nn
   from dataclay import Client, DataClayObject

   # Local application imports
   from icos_fl.models.lstm import LSTMModel
   from icos_fl.utils.colors import BGRN, WHT, paint

Avoid wildcard imports:

.. code-block:: python

   # Good
   from torch.nn import LSTM, Linear, MSELoss

   # Bad
   from torch.nn import *

Line Length
-----------

Maximum line length is 99 characters. For long lines:

- Use parentheses for line continuation in expressions
- Use backslashes only when necessary (prefer parentheses)
- Break lines before operators

.. code-block:: python

   # Good
   long_expression = (
       first_variable + second_variable + third_variable
       + fourth_variable + fifth_variable
   )

   # Also good
   def long_function_name(
       arg1: str,
       arg2: int,
       arg3: Optional[Dict[str, Any]] = None,
   ) -> None:
       pass

Whitespace
----------

Follow these whitespace conventions:

* Use 4 spaces for indentation (no tabs)
* Add two blank lines before top-level classes and functions
* Add one blank line before method definitions
* Add blank lines to separate logical sections of code

Comments
--------

* Use comments sparingly and only when needed
* Prefer docstrings for public functions and classes
* Use comments to explain complex logic or non-obvious decisions
* Keep comments up-to-date with code changes

.. code-block:: python

   # Good: explains the purpose of complex code
   # Use recursive approach for multi-step prediction to simulate real forecasting
   for _ in range(forecast_steps):
       next_pred = model(curr_seq)
       predictions.append(next_pred.item())
       curr_seq = torch.cat((curr_seq[:, 1:], next_pred.unsqueeze(0).unsqueeze(0)), dim=1)

   # Bad: states the obvious
   # Increment the counter
   counter += 1

Type Annotations
----------------

* Use type annotations for all function parameters and return values
* Use Optional for parameters that can be None
* Use Union for parameters that can have multiple types
* Use concrete types (e.g., List[int] instead of list)
* Use TypeVar for generic types

.. code-block:: python

   def create_data_loaders(
       df: pd.DataFrame,
       time_step: Optional[int] = None,
       batch_size: Optional[int] = None,
       train_ratio: Optional[float] = None,
   ) -> Tuple[DataLoader, DataLoader, TimeSeriesDataset, TimeSeriesDataset]:
       """Create data loaders for training and validation."""
       # ...

Exception Handling
------------------

* Be specific about exceptions you catch
* Use multiple except blocks for different exceptions
* Provide context in exception messages
* Avoid bare except statements

.. code-block:: python

   # Good
   try:
       data = self.time_series_data.get_dataframe()
   except DataClayException as e:
       raise DataClayException(f"Failed to retrieve time series data: {e}") from e
   except TimeoutError as e:
       raise TimeoutError(f"Timed out waiting for data: {e}") from e

   # Bad
   try:
       data = self.time_series_data.get_dataframe()
   except Exception as e:
       raise Exception(f"Error: {e}")

String Formatting
-----------------

* Use f-strings for string interpolation
* Use triple quotes for multi-line strings
* Use r-strings for raw strings (especially regex)

.. code-block:: python

   # Good
   message = f"Processing {len(df)} rows of data for metric: {metric}"

   docstring = """
   This is a multi-line docstring.
   It explains the purpose of this function.
   """

   pattern = r"^\d{4}-\d{2}-\d{2}$"  # YYYY-MM-DD

   # Bad
   message = "Processing " + str(len(df)) + " rows of data for metric: " + metric

Object-Oriented Programming
---------------------------

* Follow the Single Responsibility Principle
* Use composition over inheritance
* Make attributes private when appropriate
* Use properties for computed attributes
* Initialize all instance variables in __init__

.. code-block:: python

   class Processor:
       def __init__(
           self,
           time_step: int,
           metric: str,
           batch_size: int = 64,
           train_ratio: float = 0.8,
       ) -> None:
           self.time_step = time_step
           self.metric = metric
           self.batch_size = batch_size
           self.train_ratio = train_ratio

       @property
       def sequence_length(self) -> int:
           """Return the length of sequences created by this processor."""
           return self.time_step

Performance Considerations
--------------------------

* Use built-in functions and methods when possible
* Use vectorized operations for NumPy and pandas
* Avoid unnecessary computation in loops
* Use generators for large data processing
* Profile code before premature optimization

.. code-block:: python

   # Good: vectorized operations
   normalized_data = (df - df.mean()) / df.std()

   # Bad: inefficient loop
   normalized_data = pd.DataFrame(index=df.index)
   for col in df.columns:
       mean = df[col].mean()
       std = df[col].std()
       normalized_data[col] = [(x - mean) / std for x in df[col]]
