=======
Testing
=======

This page describes how to write and run tests for ICOS-FL.

Testing Framework
-----------------

ICOS-FL uses pytest as its testing framework. Tests are located in the ``tests/`` directory and follow the naming convention ``test_*.py``.

Setting Up Testing Environment
------------------------------

1. Install testing dependencies:

   .. code-block:: bash

      pip install -e ".[test]"

2. Run the test suite:

   .. code-block:: bash

      pytest

Test Structure
--------------

Tests are organized by the module they test:

.. code-block:: text

   tests/
   ├── test_icos_fl.py      # Basic project tests
   ├── test_fetcher.py      # Tests for utils.fetcher
   ├── test_lstm.py         # Tests for models.lstm
   └── test_processor.py    # Tests for utils.processor

Test Categories
---------------

ICOS-FL has several types of tests:

Unit Tests
~~~~~~~~~~

Unit tests verify individual components in isolation:

.. code-block:: python

   def test_normalize_data():
       """Test data normalization function."""
       processor = Processor(time_step=10, metric="cpu_usage")
       df = pd.DataFrame({
           "col1": [1, 2, 3, 4, 5],
           "col2": [5, 4, 3, 2, 1]
       })
       normalized = processor._normalize_data(df)

       # Check that data is normalized (mean=0, std=1)
       assert abs(normalized["col1"].mean()) < 1e-10
       assert abs(normalized["col2"].mean()) < 1e-10
       assert abs(normalized["col1"].std() - 1.0) < 1e-10
       assert abs(normalized["col2"].std() - 1.0) < 1e-10

Integration Tests
~~~~~~~~~~~~~~~~~

Integration tests verify that components work together:

.. code-block:: python

   def test_fetcher_processor_integration():
       """Test that Fetcher and Processor work together correctly."""
       fetcher = Fetcher()
       df = fetcher.fetch_data(timeout=10)

       processor = Processor(time_step=10, metric="cpu_usage")
       train_loader, val_loader, _, _ = processor.create_data_loaders(df)

       # Verify dataloaders are created correctly
       assert train_loader is not None
       assert val_loader is not None
       assert len(train_loader) > 0
       assert len(val_loader) > 0

Functional Tests
~~~~~~~~~~~~~~~~

Functional tests verify end-to-end behavior:

.. code-block:: python

   def test_model_training():
       """Test full model training cycle."""
       # Get data
       fetcher = Fetcher()
       df = fetcher.fetch_data(timeout=10)

       # Process data
       processor = Processor(time_step=10, metric="cpu_usage")
       train_loader, val_loader, _, _ = processor.create_data_loaders(df)

       # Initialize model
       model = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

       # Train model
       train_loss = train(model, train_loader, epochs=2, lr=0.01)

       # Evaluate model
       val_loss = test(model, val_loader, torch.device("cpu"))

       # Verify training was successful
       assert train_loss > 0
       assert val_loss > 0

Test Fixtures
-------------

Use pytest fixtures for common setup:

.. code-block:: python

   @pytest.fixture
   def sample_dataframe():
       """Create a test DataFrame with synthetic data."""
       return pd.DataFrame({
           "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="h"),
           "cpu_usage": np.random.uniform(0.1, 0.9, 100),
           "memory_usage": np.random.uniform(0.2, 0.8, 100),
           "power_consumption": np.random.uniform(10, 100, 100)
       })

   @pytest.fixture
   def processor():
       """Create a Processor instance for testing."""
       return Processor(
           time_step=10,
           metric="cpu_usage",
           batch_size=16,
           train_ratio=0.8,
           device=torch.device("cpu")
       )

   def test_create_data_loaders(sample_dataframe, processor):
       """Test creating data loaders from a DataFrame."""
       train_loader, val_loader, _, _ = processor.create_data_loaders(sample_dataframe)
       assert len(train_loader) > 0
       assert len(val_loader) > 0

Mocking
-------

Use mocking to test components with external dependencies:

.. code-block:: python

   @patch("icos_fl.utils.fetcher.Client")
   def test_fetcher_connection_error(mock_client):
       """Test handling of connection errors."""
       # Configure the mock to raise an exception
       mock_client.return_value.start.side_effect = DataClayException("Connection error")

       # Verify that the Fetcher handles the exception
       with pytest.raises(DataClayException):
           fetcher = Fetcher()

Testing Model Components
------------------------

For testing LSTM models:

.. code-block:: python

   def test_lstm_forward_pass():
       """Test the forward pass of the LSTM model."""
       # Create model
       model = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

       # Create input tensor: [batch_size, 1, time_step]
       batch_size = 8
       input_tensor = torch.randn(batch_size, 1, 10)

       # Perform forward pass
       with torch.no_grad():
           output = model(input_tensor)

       # Verify output shape: [batch_size, 1]
       assert output.shape == (batch_size, 1)

Testing Federated Components
----------------------------

For testing federated learning components, use Flower's testing utilities:

.. code-block:: python

   from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

   def test_client_fit():
       """Test the fit method of the client."""
       # Create model and data
       model = LSTMModel(hidden_layer_size=10, time_step=10, num_layers=1)

       # Create synthetic data
       train_dataset = SyntheticDataset(size=100)
       val_dataset = SyntheticDataset(size=50)

       train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

       # Create client
       client = IcosClient(
           client_id="test-client",
           client_state=None,
           model=model,
           trainloader=train_loader,
           valloader=val_loader,
           local_epochs=2,
           metric="cpu_usage"
       )

       # Create parameters
       params = ndarrays_to_parameters(get_weights(model))

       # Test fit method
       fit_result = client.fit(parameters_to_ndarrays(params), {"server_round": 1})

       # Verify results
       assert fit_result is not None
       assert len(fit_result) == 3
       assert isinstance(fit_result[0], list)  # parameters
       assert isinstance(fit_result[1], int)   # num examples
       assert isinstance(fit_result[2], dict)  # metrics

Code Coverage
-------------

To run tests with coverage:

.. code-block:: bash

   pytest --cov=icos_fl tests/

To generate a coverage report:

.. code-block:: bash

   pytest --cov=icos_fl --cov-report=html tests/

This creates an HTML coverage report in the ``htmlcov/`` directory.

Continuous Integration
----------------------

ICOS-FL uses GitHub Actions for continuous integration. Tests are automatically run for each pull request and commit to the main branch.

The CI workflow:

1. Runs linting and formatting checks
2. Runs the full test suite
3. Generates a test coverage report
4. Builds and verifies the documentation

Adding New Tests
----------------

When adding new features, always include tests:

1. Create a new test file or add to an existing one
2. Include both positive and negative test cases
3. Cover normal operation and error conditions
4. Test edge cases and boundary conditions
5. Verify all public interfaces

Best Practices
--------------

1. **Keep tests focused**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should describe what they test
3. **Isolate tests**: Tests should not depend on each other
4. **Avoid test duplication**: Use fixtures and parameterization
5. **Clean up after tests**: Ensure tests don't leave behind artifacts
6. **Test error conditions**: Verify exceptions are raised correctly
7. **Keep tests fast**: Avoid unnecessary computation
8. **Mock external services**: Don't depend on external services in tests

Running Specific Tests
----------------------

To run a specific test:

.. code-block:: bash

   pytest tests/test_fetcher.py::test_fetch_data

To run tests matching a pattern:

.. code-block:: bash

   pytest -k "fetcher"

To run tests with verbose output:

.. code-block:: bash

   pytest -v
