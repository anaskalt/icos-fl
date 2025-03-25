"""Client implementation for ICOS-FL using Flower's ClientApp architecture.

This module defines the client-side components of the federated learning system,
including local model training and evaluation within the Flower framework.
"""

from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, RecordSet

from icos_fl.models.lstm import LSTMModel, get_weights, set_weights, test, train
from icos_fl.utils.logger import Logger
from icos_fl.utils.processor import Processor

# TODO: Uncomment the following import when fetcher.py is available:
# from icos_fl.utils.fetcher import Fetcher  # noqa: ERA001


# Configure logger
logger = Logger(useconsole=True, usecolor=True)


class IcosClient(NumPyClient):
    """Client implementation for ICOS FL with time series prediction.

    This client trains an LSTM model for time series prediction
    in a federated learning setting. It handles local model training,
    evaluation, and parameter management as part of the federated
    learning process.
    """

    def __init__(
        self,
        client_id: Union[int, str],
        client_state: RecordSet,
        model: LSTMModel,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        local_epochs: int,
        metric: str,
    ) -> None:
        """Initialize the ICOS client.

        Args:
            client_id: Unique identifier for this client
            client_state: RecordSet to maintain state across rounds
            model: The LSTM model to train
            trainloader: DataLoader for training data
            valloader: DataLoader for validation data
            local_epochs: Number of local training epochs per FL round
            metric: Name of the metric being predicted
        """
        self.client_id = client_id
        self.client_state = client_state
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.metric = metric

        # Set device for computation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        init_message = f"Initialized ICOS Client {self.client_id} for metric {self.metric}"
        device_message = f"Using device: {self.device}"
        logger.info(init_message)
        logger.info(device_message)

    def get_parameters(self, config: Dict[str, Any]) -> List[NDArrays]:
        """Get model parameters as a list of NumPy arrays.

        Args:
            config: Configuration parameters

        Returns:
            List of model parameter arrays
        """
        return get_weights(self.model)

    def set_parameters(self, parameters: List[NDArrays]) -> None:
        """Set model parameters from a list of NumPy arrays.

        Args:
            parameters: List of model parameter arrays
        """
        set_weights(self.model, parameters)

    def fit(
        self, parameters: List[NDArrays], config: Dict[str, Any]
    ) -> Tuple[List[NDArrays], int, Dict[str, Any]]:
        """Train the model on the local dataset.

        Args:
            parameters: Current global model parameters
            config: Configuration parameters for training

        Returns:
            Tuple containing (updated model parameters, number of training samples, metrics)
        """
        # Get current round from config
        server_round = int(config.get("server_round", 1))

        # Get learning rate from config or use default
        lr = float(config.get("lr", 0.001))

        fit_start_msg = f"Client {self.client_id} starting fit() for round {server_round}"
        logger.info(fit_start_msg)

        # Update local model with global parameters
        self.set_parameters(parameters)

        # Train the model
        train_loss = train(
            model=self.model, train_dataloader=self.trainloader, epochs=self.local_epochs, lr=lr
        )

        # Log training completion
        fit_complete_msg = f"Client {self.client_id} completed fit() for round {server_round}"
        loss_msg = f"Training loss: {train_loss:.6f}"
        logger.info(fit_complete_msg)
        logger.info(loss_msg)

        # Return updated model parameters and metrics
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss), "client_id": self.client_id, "metric": self.metric},
        )

    def evaluate(
        self, parameters: List[NDArrays], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on the local validation dataset.

        Args:
            parameters: Current global model parameters
            config: Configuration parameters for evaluation

        Returns:
            Tuple containing (loss, number of validation samples, metrics)
        """
        # Get current round from config
        server_round = int(config.get("server_round", 1))

        eval_start_msg = f"Client {self.client_id} starting evaluate() for round {server_round}"
        logger.info(eval_start_msg)

        # Update local model with global parameters
        self.set_parameters(parameters)

        # Evaluate the model
        val_loss = test(model=self.model, test_dataloader=self.valloader, device=self.device)

        # Log evaluation completion
        eval_complete_msg = (
            f"Client {self.client_id} completed evaluate() for round {server_round}"
        )
        val_loss_msg = f"Validation loss: {val_loss:.6f}"
        logger.info(eval_complete_msg)
        logger.info(val_loss_msg)

        # Return evaluation metrics
        return (
            float(val_loss),
            len(self.valloader.dataset),
            {"val_loss": float(val_loss), "client_id": self.client_id, "metric": self.metric},
        )


def fetch_data(metric: str) -> pd.DataFrame:
    """Temporary placeholder for fetcher.py functionality.

    This function will be replaced with the actual implementation
    when fetcher.py is available.

    Args:
        metric: The metric to fetch data for

    Returns:
        DataFrame with time series data for the requested metric
    """
    logger.warning(
        "Using placeholder data fetching function. Replace with actual fetcher when available."
    )

    # In the actual implementation, this will call:
    # return fetcher.fetch_data(metric)  # noqa: ERA001

    # For now, return None to indicate that actual data fetching is not available
    return None


def client_fn(context: Context) -> NumPyClient:
    """Create and return a Flower client instance.

    This function is used by the ClientApp to create a client instance
    for each node in the federated learning system. It extracts configuration
    from the context, initializes the model, and prepares data loaders
    for training and evaluation.

    Args:
        context: Flower client context with configuration and node info

    Returns:
        Instantiated NumPyClient
    """
    # Extract client ID from context
    client_id = context.node_config.get("cid", context.node_id)

    # Extract configuration from context
    metric = context.run_config.get("metric", "cpu_consumption")
    local_epochs = int(context.run_config.get("local-epochs", 1))

    # Model hyperparameters
    hidden_layer_size = int(context.run_config.get("hidden-layer-size", 10))
    time_step = int(context.run_config.get("time-step", 10))
    num_layers = int(context.run_config.get("num-layers", 1))
    batch_size = int(context.run_config.get("batch-size", 64))
    train_test_split = float(context.run_config.get("train-test-split", 0.8))

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fetch data
    # TODO: This will be replaced with an actual call to fetcher.py when available
    # df = fetcher.fetch_data(metric)  # noqa: ERA001
    df = fetch_data(metric)

    # Initialize train and validation dataloaders
    train_dataloader = None
    val_dataloader = None

    if df is not None:
        # Create processor for data preparation
        processor = Processor(
            time_step=time_step,
            metric=metric,
            batch_size=batch_size,
            train_ratio=train_test_split,
            device=device,
        )

        # Create dataloaders
        train_dataloader, val_dataloader, _, _ = processor.create_data_loaders(df)
        success_msg = f"Client {client_id}: Successfully created data loaders for {metric}"
        logger.info(success_msg)
    else:
        no_data_msg = f"Client {client_id}: No data available - fetcher.py integration required"
        logger.warning(no_data_msg)

    # Initialize model
    model = LSTMModel(
        hidden_layer_size=hidden_layer_size, time_step=time_step, num_layers=num_layers
    ).to(device)

    # Create and return client instance
    client = IcosClient(
        client_id=client_id,
        client_state=context.state,
        model=model,
        trainloader=train_dataloader,
        valloader=val_dataloader,
        local_epochs=local_epochs,
        metric=metric,
    )

    return client


# Define the Flower ClientApp
app = ClientApp(client_fn=client_fn)
