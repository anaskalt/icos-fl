"""Server implementation for ICOS-FL using Flower's ServerApp architecture.

This module defines the server setup for the federated learning system,
initializing and configuring the Flower ServerApp with appropriate strategy
for centralized model aggregation and evaluation.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from icos_fl.models.lstm import LSTMModel, get_weights, set_weights
from icos_fl.server.strategy import (
    CustomFedAvg,
    evaluate_metrics_aggregation,
    train_metrics_aggregation,
)
from icos_fl.utils.logger import Logger

# TODO: Uncomment when fetcher.py is available:
# from icos_fl.utils.fetcher import Fetcher  # noqa: ERA001


# Configure logger
logger = Logger(useconsole=True, usecolor=True)


def gen_evaluate_fn(
    model: LSTMModel, metric: str, time_step: int, device: torch.device
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate a centralized evaluation function.

    This function creates a callable that can evaluate the global model
    on a centralized test dataset.

    Args:
        model: The LSTM model to evaluate
        metric: The metric being predicted
        time_step: Window size for time series prediction
        device: Device to run evaluation on

    Returns:
        Function that takes (round, parameters, config) and returns (loss, metrics)
    """

    # This function will be defined properly once fetcher.py is available
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on centralized test set."""
        eval_msg = f"Running centralized evaluation for round {server_round}"
        logger.info(eval_msg)

        # Set the model weights
        ndarrays = ndarrays_to_parameters(parameters)
        set_weights(model, ndarrays)

        # TODO: Fetch test dataset using Fetcher when available
        # df = Fetcher().fetch_data(metric)  # noqa: ERA001

        # For now, we'll return placeholder values
        logger.warning("Centralized evaluation not fully implemented - waiting for fetcher.py")

        # This will be replaced with actual evaluation when the Fetcher is available:
        # processor = Processor(time_step=time_step, metric=metric, device=device)  # noqa: ERA001
        # _, val_dataloader, _, _ = processor.create_data_loaders(df)  # noqa: ERA001
        # val_loss = test(model, val_dataloader, device)  # noqa: ERA001
        # return val_loss, {"centralized_loss": val_loss}  # noqa: ERA001

        # Return placeholder values for now
        return 0.0, {"centralized_loss": 0.0}

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    """Create and return a Flower server instance.

    This function is used by the ServerApp to create server components
    for the federated learning system.

    Args:
        context: Flower server context

    Returns:
        Instantiated ServerAppComponents
    """
    # Extract configuration from context
    num_rounds = int(context.run_config.get("num-server-rounds", 10))
    fraction_fit = float(context.run_config.get("fraction-fit", 0.5))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 0.5))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 2))
    min_evaluate_clients = int(context.run_config.get("min-evaluate-clients", 2))
    min_available_clients = int(context.run_config.get("min-available-clients", 2))
    metric = context.run_config.get("metric", "cpu_consumption")
    use_wandb = context.run_config.get("use-wandb", True)

    # Create model parameters
    hidden_layer_size = int(context.run_config.get("hidden-layer-size", 10))
    time_step = int(context.run_config.get("time-step", 10))
    num_layers = int(context.run_config.get("num-layers", 1))

    # Create result directory
    save_dir = os.path.join("model", metric)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model
    device = torch.device(context.run_config.get("server-device", "cpu"))
    model = LSTMModel(
        hidden_layer_size=hidden_layer_size, time_step=time_step, num_layers=num_layers
    ).to(device)

    # Get initial model parameters
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    # Create strategy
    strategy = CustomFedAvg(
        # Custom parameters
        run_config=context.run_config,
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
        fit_metrics_aggregation_fn=train_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        # Centralized evaluation function
        # Will be properly implemented once fetcher.py is available
        evaluate_fn=gen_evaluate_fn(model, metric, time_step, device),
    )

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    # Log server configuration
    start_msg = f"Starting server for metric {metric}"
    rounds_msg = f"Number of rounds: {num_rounds}"
    fraction_msg = f"Fraction fit: {fraction_fit}, Fraction evaluate: {fraction_evaluate}"
    clients_msg = (
        f"Minimum clients - fit: {min_fit_clients}, evaluate: {min_evaluate_clients}, "
        f"available: {min_available_clients}"
    )
    save_msg = f"Model saved to: {save_dir}"

    logger.info(start_msg)
    logger.info(rounds_msg)
    logger.info(fraction_msg)
    logger.info(clients_msg)
    logger.info(save_msg)

    return ServerAppComponents(strategy=strategy, config=config)


# Define the Flower ServerApp
app = ServerApp(server_fn=server_fn)
