=============
Documentation
=============

This page provides guidelines for contributing to ICOS-FL's documentation.

Documentation Structure
-----------------------

ICOS-FL's documentation follows the Diátaxis framework, which organizes content into four categories:

1. **Tutorials** (Introduction): Learning-oriented content that helps new users get started
2. **How-To Guides**: Task-oriented guides for solving specific problems
3. **Explanation**: Understanding-oriented content that provides background and context
4. **Reference**: Information-oriented technical descriptions of the machinery

File Organization
-----------------

The documentation is organized into these main sections:

.. code-block:: text

   docs/
   ├── source/
   │   ├── _static/             # Static assets (images, CSS)
   │   ├── _templates/          # Custom templates
   │   ├── introduction/        # Getting started tutorials
   │   ├── how_to/              # Task-based guides
   │   ├── explanation/         # Concept explanations
   │   ├── reference/           # API reference
   │   ├── contributing/        # Contribution guidelines
   │   ├── conf.py              # Sphinx configuration
   │   └── index.rst            # Main index page
   ├── Makefile                 # Make commands for building docs
   └── requirements.txt         # Documentation dependencies

Writing Style
-------------

When writing documentation, follow these style guidelines:

1. **Use present tense**: Write "this function returns" rather than "this function will return"
2. **Active voice**: Use "ICOS-FL provides" instead of "is provided by ICOS-FL"
3. **Second person**: Address the reader as "you" rather than "the user"
4. **Simple language**: Avoid jargon and complex terms where possible
5. **Concise**: Be direct and to the point
6. **Consistent terminology**: Use the same terms throughout the documentation

reStructuredText Syntax
-----------------------

ICOS-FL documentation uses reStructuredText (reST) format. Here are some common syntax elements:

Headings
~~~~~~~~

Use consistent heading levels:

.. code-block:: rst

   =============
   Section Title
   =============

   Subsection Title
   ---------------

   Subsubsection Title
   ~~~~~~~~~~~~~~~~~~

   Paragraph Title
   """"""""""""""

Inline Formatting
~~~~~~~~~~~~~~~~~

- *Italic*: ``*italic*``
- **Bold**: ``**bold**``
- ``Code``: ````code````
- Links: ```Link text <https://example.com>`_``

Code Blocks
~~~~~~~~~~~

For code examples:

.. code-block:: rst

   .. code-block:: python

      def example_function():
          """This is an example function."""
          return True

For console commands:

.. code-block:: rst

   .. code-block:: bash

      $ python -m build

Lists
~~~~~

Bulleted lists:

.. code-block:: rst

   * Item 1
   * Item 2
     * Subitem 2.1
     * Subitem 2.2
   * Item 3

Numbered lists:

.. code-block:: rst

   1. First item
   2. Second item
   3. Third item

Definition lists:

.. code-block:: rst

   Term 1
       Definition 1

   Term 2
       Definition 2

Images
~~~~~~

.. code-block:: rst

   .. figure:: ../../_static/images/architecture_overview.png
      :alt: ICOS-FL Architecture Overview
      :align: center

      Caption text here

Tables
~~~~~~

.. code-block:: rst

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Header 1
        - Header 2
      * - Cell 1
        - Cell 2
      * - Cell 3
        - Cell 4

Notes and Warnings
~~~~~~~~~~~~~~~~~~

.. code-block:: rst

   .. note::
      This is a note.

   .. warning::
      This is a warning.

   .. tip::
      This is a tip.

   .. important::
      This is important information.

Cross-References
~~~~~~~~~~~~~~~~

To reference other parts of the documentation:

.. code-block:: rst

   :doc:`/how_to/deployment/docker_setup`

   :ref:`custom-models`

Building Documentation
----------------------

To build the documentation locally:

1. Install documentation dependencies:

   .. code-block:: bash

      pip install -r docs/requirements.txt

2. Build HTML documentation:

   .. code-block:: bash

      cd docs
      make html

3. View the documentation:

   Open ``docs/build/html/index.html`` in your web browser.

Adding New Documentation
------------------------

To add new documentation:

1. Create a new `.rst` file in the appropriate directory
2. Add the file to the corresponding index (toctree)
3. Include proper headings, formatting, and cross-references
4. Build the documentation to verify it renders correctly
5. Submit a pull request with your changes

API Documentation
-----------------

When documenting API components:

1. Ensure all public classes, methods, and functions have proper docstrings
2. Use Google-style docstrings for consistency
3. Include type information and parameter descriptions
4. Document exceptions that may be raised
5. Provide usage examples where appropriate

Example:

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

Images and Diagrams
-------------------

For adding images and diagrams:

1. Place image files in ``docs/source/_static/images/``
2. Use clear, high-contrast images
3. Include alt text for accessibility
4. Add captions to explain the image contents
5. Optimize images for web (compress when possible)
6. Use SVG format for diagrams when available

Documentation Review Process
----------------------------

All documentation changes follow the same review process as code:

1. Create a PR with your documentation changes
2. Ensure documentation builds without warnings
3. Request review from relevant maintainers
4. Address feedback from reviews
5. Once approved, a maintainer will merge your PR
