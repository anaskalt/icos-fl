"""Server implementation for ICOS-FL using Flower's ServerApp architecture.

This module defines the server setup for the federated learning system,
initializing and configuring the Flower ServerApp with appropriate strategy
for centralized model aggregation, evaluation and multi-backend persistence.
"""

import logging
import os
from logging import CRITICAL, INFO, WARN
from typing import Callable, Dict, Optional, Tuple

import torch
from dataclay.exceptions import DataClayException
from flwr.common import (
    Context,
    NDArrays,
    Scalar,
    logger,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from icos_fl.models.lstm import LSTMModel, get_weights, set_weights, test
from icos_fl.server.strategy import (
    CustomFedAvg,
    evaluate_metrics_aggregation,
    train_metrics_aggregation,
)
from icos_fl.utils.fetcher import Fetcher
from icos_fl.utils.logo import print_server_banner
from icos_fl.utils.processor import Processor

logging.getLogger("flwr").propagate = False

# Display server banner
print_server_banner()


def gen_evaluate_fn(
    model: LSTMModel,
    metric: str,
    time_step: int,
    batch_size: int,
    device: torch.device,
    dataclay_host: str,
    dataclay_dataset: str,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate a centralized evaluation function.

    This function creates a callable that can evaluate the global model
    on a centralized test dataset.

    Args:
        model: The LSTM model to evaluate
        metric: The metric being predicted
        time_step: Window size for time series prediction
        batch_size: Batch size for data loaders
        device: Device to run evaluation on
        dataclay_host: DataClay proxy host address
        dataclay_dataset: DataClay dataset name

    Returns:
        Function that takes (round, parameters, config) and returns (loss, metrics)
    """

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on centralized test set."""
        eval_msg = f"Running centralized evaluation for round {server_round}"
        logger.log(INFO, eval_msg)

        try:
            # Set the model weights
            model_params = parameters_to_ndarrays(parameters)
            set_weights(model, model_params)

            df = None

            # Create fetcher with the same DataClay configuration
            fetcher = Fetcher(proxy_host=dataclay_host, dataset=dataclay_dataset)

            # Fetch test dataset using Fetcher
            try:
                df = fetcher.fetch_data(timeout=60)
            except Exception as e:  # noqa: BLE001
                error_msg = f"Error fetching data: {e}"
                logger.log(CRITICAL, error_msg)
                return None
            finally:
                # Clean up fetcher after use
                try:
                    fetcher._disconnect()
                except Exception as e:  # noqa: BLE001
                    # Log disconnection errors at warning level for debugging
                    disconnect_warning_msg = f"Warning during fetcher disconnect: {e}"
                    logger.log(WARN, disconnect_warning_msg)

            if df is None or len(df) == 0:
                no_data_msg = "No data available for centralized evaluation"
                logger.log(WARN, no_data_msg)
                return None

            # Create processor for data preparation
            processor = Processor(time_step=time_step, metric=metric, device=device)

            # Create dataloaders with emphasis on validation data
            # Using a low train_ratio to prioritize evaluation data
            _, val_dataloader, _, _ = processor.create_data_loaders(
                df, train_ratio=0.2, batch_size=batch_size
            )

            # Evaluate the model
            val_loss = test(model, val_dataloader, device)

            # Calculate accuracy
            accuracy = max(0.0, 1.0 - val_loss)

            eval_result_msg = (
                f"Centralized evaluation loss: {val_loss:.6f}, "
                f"Centralized evaluation accuracy: {accuracy:.6f}"
            )
            logger.log(INFO, eval_result_msg)

            return val_loss, {"centralized_accuracy": accuracy}
        except DataClayException as e:
            # Handle specific exceptions from DataClay
            error_msg = f"DataClay error during centralized evaluation: {e}"
            logger.log(CRITICAL, error_msg)
            return None
        except TimeoutError as e:
            # Handle timeout errors
            timeout_msg = f"Timeout during centralized evaluation: {e}"
            logger.log(CRITICAL, timeout_msg)
            return None
        except Exception as e:  # noqa: BLE001
            # Handle any other exceptions
            error_msg = f"Unexpected error during centralized evaluation: {e}"
            logger.log(CRITICAL, error_msg)
            return None

    return evaluate


def create_on_fit_config_fn(learning_rate: float) -> Callable[[int], Dict[str, Scalar]]:
    """Create an on_fit_config function with the specified learning rate.

    Args:
        learning_rate: Learning rate to use in the config

    Returns:
        A function that creates client configs with this learning rate
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration dict for each round."""
        config = {
            "server_round": server_round,
            "lr": learning_rate,
        }
        return config

    return on_fit_config


def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration dict for each round.

    Args:
        server_round: Current federated learning round

    Returns:
        Configuration dictionary with server round
    """
    return {
        "server_round": server_round,
    }


def create_model_directory(base_dir: str, metric: str) -> str:
    """Create and return the model directory path when local storage is enabled.

    This function attempts to create a directory for storing model artifacts.
    If the requested directory cannot be created, it falls back to /tmp.

    Args:
        base_dir: Base directory path for storing models
        metric: Metric name to use for the subdirectory

    Returns:
        Path to the created directory where models will be stored
    """
    # Construct primary directory path
    save_dir = os.path.join(base_dir, metric)

    try:
        # Try to create the directory
        os.makedirs(save_dir, exist_ok=True)
        created_msg = f"Model directory prepared: {save_dir}"
        logger.log(INFO, created_msg)
        return save_dir
    except (PermissionError, OSError) as e:
        # Fall back to /tmp
        import tempfile

        error_msg = f"Cannot create directory at {save_dir}: {e}"
        fallback_msg = "Using temporary directory for model storage"
        logger.log(WARN, error_msg)
        logger.log(INFO, fallback_msg)

        # Use a temporary directory as fallback
        temp_base = tempfile.gettempdir()
        tmp_dir = os.path.join(temp_base, f"icos-fl-{metric}")
        os.makedirs(tmp_dir, exist_ok=True)

        tmp_msg = f"Temporary directory prepared: {tmp_dir}"
        logger.log(INFO, tmp_msg)
        return tmp_dir


def server_fn(context: Context) -> ServerAppComponents:
    """Create and return a Flower server instance with multi-backend storage support.

    This function serves as the primary entry point for ServerApp initialization,
    configuring the federated learning system with flexible storage options.

    Key Responsibilities:
    - Extract and validate configuration parameters from context
    - Initialize model directory structure when local storage is enabled
    - Configure CustomFedAvg strategy with appropriate storage backends
    - Set up centralized evaluation functionality

    Storage Backend Configuration:
    - save-local: Enable local filesystem storage (default: True)
    - save-dataclay: Enable DataClay distributed storage (default: False)
    - When both are False, defaults to local storage for reliability

    Args:
        context: Flower server context containing complete run configuration

    Returns:
        ServerAppComponents with configured strategy and server config
    """
    # Extract core FL configuration from context
    num_rounds = int(context.run_config.get("num-server-rounds", 10))
    fraction_fit = float(context.run_config.get("fraction-fit", 0.5))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 0.5))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 2))
    min_evaluate_clients = int(context.run_config.get("min-evaluate-clients", 2))
    min_available_clients = int(context.run_config.get("min-available-clients", 2))
    metric = context.run_config.get("metric", "cpu_usage")
    batch_size = int(context.run_config.get("batch-size", 64))
    use_wandb = context.run_config.get("use-wandb", True)

    # Extract training configuration
    learning_rate = float(context.run_config.get("learning-rate", 0.001))
    on_fit_config_fn = create_on_fit_config_fn(learning_rate)

    # Extract model architecture parameters
    hidden_layer_size = int(context.run_config.get("hidden-layer-size", 10))
    time_step = int(context.run_config.get("time-step", 10))
    num_layers = int(context.run_config.get("num-layers", 1))

    # Extract DataClay configuration
    dataclay_host = context.run_config.get("dataclay-host", "127.0.0.1")
    dataclay_dataset = context.run_config.get("dataclay-dataset", "admin")

    # Extract storage configuration
    save_local = context.run_config.get("save-local", True)

    # Get result directory from context
    result_dir = context.run_config.get("result-dir", "/app/outputs/models")

    # Only create model directory if local storage is enabled
    save_dir = None
    if save_local:
        save_dir = create_model_directory(result_dir, metric)

    # Initialize model architecture
    device = torch.device(context.run_config.get("server-device", "cpu"))
    model = LSTMModel(
        hidden_layer_size=hidden_layer_size, time_step=time_step, num_layers=num_layers
    ).to(device)

    # Get initial model parameters
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    # Create strategy with storage configuration
    # Note: CustomFedAvg will read storage options directly from run_config
    strategy = CustomFedAvg(
        # Storage configuration
        run_config=context.run_config,  # Pass entire config for storage options
        model=model,
        metric=metric,
        use_wandb=use_wandb,
        save_dir=save_dir,
        # FedAvg parameters
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config,
        fit_metrics_aggregation_fn=train_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        # Centralized evaluation function
        evaluate_fn=gen_evaluate_fn(
            model, metric, time_step, batch_size, device, dataclay_host, dataclay_dataset
        ),
    )

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    # Log server configuration for visibility
    start_msg = f"Starting server for metric {metric}"
    rounds_msg = f"Number of rounds: {num_rounds}"
    fraction_msg = f"Fraction fit: {fraction_fit}, Fraction evaluate: {fraction_evaluate}"
    clients_msg = (
        f"Minimum clients - fit: {min_fit_clients}, evaluate: {min_evaluate_clients}, "
        f"available: {min_available_clients}"
    )

    logger.log(INFO, start_msg)
    logger.log(INFO, rounds_msg)
    logger.log(INFO, fraction_msg)
    logger.log(INFO, clients_msg)

    return ServerAppComponents(strategy=strategy, config=config)


# Define the Flower ServerApp
app = ServerApp(server_fn=server_fn)
