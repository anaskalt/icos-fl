===============
Code Guidelines
===============

This page describes how to contribute code to ICOS-FL.

Setting Up Development Environment
----------------------------------

1. Fork the repository and clone it to your local machine:

   .. code-block:: bash

      git clone https://github.com/yourusername/icos-fl.git
      cd icos-fl

2. Create a virtual environment and install the package in development mode:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev,test,lint]"

3. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

This will set up linting, formatting, and type checking that will run automatically on commit.

Development Workflow
--------------------

1. **Create a Feature Branch**

   Always create a new branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make Changes**

   Implement your changes following our coding standards (detailed below).

3. **Run Tests**

   Ensure your changes don't break existing functionality:

   .. code-block:: bash

      pytest

4. **Format and Lint Code**

   Format your code and check for linting errors:

   .. code-block:: bash

      isort .
      black .
      ruff check --fix .
      mypy .

5. **Commit Changes**

   Follow the :ref:`commit-guidelines` when writing commit messages.

6. **Create a Pull Request**

   Push your changes and create a pull request on GitHub.

Coding Standards
----------------

ICOS-FL follows the following coding standards:

Python Version
~~~~~~~~~~~~~~

ICOS-FL supports Python 3.10 and newer. Ensure your code is compatible with these versions.

Imports
~~~~~~~

- Use absolute imports for external packages
- Use relative imports for internal modules
- Group imports in the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Use `isort` to automatically sort imports

Type Annotations
~~~~~~~~~~~~~~~~

All code should include type annotations:

- Use type annotations for all function parameters and return values
- Use `Optional` for parameters that can be `None`
- Use `Union` for parameters that can be multiple types
- Use generic types where applicable (e.g., `List[int]`, `Dict[str, Any]`)

Documentation
~~~~~~~~~~~~~

- All public classes, methods, and functions should have docstrings
- Follow Google-style docstring format
- Include type information in docstrings for better IDE support

Example:

.. code-block:: python

   def process_metrics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
       """Process metrics data.

       Args:
           df: DataFrame containing raw metrics data
           columns: List of column names to process

       Returns:
           Processed DataFrame with transformed metrics

       Raises:
           ValueError: If any column in columns is not present in df
       """

Error Handling
~~~~~~~~~~~~~~

- Use specific exception types rather than generic `Exception`
- Handle exceptions at the appropriate level
- Document expected exceptions in docstrings
- Use context managers for resource management

Testing
~~~~~~~

- Write unit tests for all new functionality
- Use pytest fixtures for common test setup
- Aim for high test coverage, especially for critical components
- Include both positive and negative test cases

.. _commit-guidelines:

Commit Guidelines
-----------------

We follow conventional commits for commit messages:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

Example commit message:

.. code-block:: text

   feat(client): add support for multi-step prediction

   Implement recursive multi-step prediction in the LSTM model.
   This allows the model to forecast further into the future by
   recursively feeding predictions back into the model.

   Closes #123

Pull Request Process
--------------------

1. **Create a Pull Request (PR)** with a clear title and description
2. **Link to any related issues** using keywords like "Fixes #123" or "Relates to #456"
3. **Ensure all CI checks pass**
4. **Request a review** from relevant maintainers
5. **Address feedback** from code review
6. **Update documentation** if your changes affect user-facing features
7. Once approved, a maintainer will merge your PR

Pull requests should be focused on a single feature or bug fix to make review easier.

Versioning
----------

ICOS-FL follows Semantic Versioning (SemVer):

- **MAJOR** version for incompatible API changes
- **MINOR** version for functionality added in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

Version updates are handled by maintainers. Contributors should not modify version numbers.

Code of Conduct
---------------

All contributors are expected to follow our Code of Conduct. Please be respectful and considerate of others when contributing to the project.
