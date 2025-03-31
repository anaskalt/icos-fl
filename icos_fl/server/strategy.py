"""Strategy implementation for ICOS-FL federated learning.

This module defines custom strategies for federated learning aggregation,
including model update aggregation, metrics tracking, and checkpoint
management.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Metrics, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import wandb
from icos_fl.models.lstm import LSTMModel, set_weights
from icos_fl.utils.logger import Logger

# Configure logger
logger = Logger()


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy for ICOS-FL.

    Extends the standard FedAvg strategy with additional functionality:
    - Tracks and logs metrics using Weights & Biases
    - Saves model checkpoints

    Args:
        *args: Variable length argument list for FedAvg parent class
        **kwargs: Arbitrary keyword arguments for FedAvg parent class
        run_config: Configuration dictionary from context
        model: The LSTM model used
        metric: The metric being predicted
        use_wandb: Whether to use Weights & Biases for logging
        save_dir: Directory for saving model checkpoints
    """

    def __init__(
        self,
        run_config: Dict[str, Any],
        model: LSTMModel,
        metric: str,
        use_wandb: bool = True,
        save_dir: str = "model",
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the custom FedAvg strategy."""
        super().__init__(*args, **kwargs)

        self.run_config = run_config
        self.model = model
        self.metric = metric
        self.use_wandb = use_wandb
        self.save_dir = save_dir

        # Create directory for saving results
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize WandB if enabled
        if self.use_wandb:
            self._init_wandb()

        # Keep track of best metrics
        self.best_metrics = {"loss": float("inf")}

        init_msg = f"Initialized CustomFedAvg strategy for metric {metric}"
        save_msg = f"Saving models to {save_dir}"
        logger.info(init_msg)
        logger.info(save_msg)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases project."""
        wandb.init(project="ICOS-FL", name=f"{self.metric}-ServerApp", config=self.run_config)  # type: ignore

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients and update the global model.

        Args:
            server_round: Current round of federated learning
            results: List of tuples of (client, fit result)
            failures: List of failures that occurred during fitting

        Returns:
            Tuple of (new global parameters, aggregation metrics)
        """
        # Log round information
        agg_msg = f"Aggregating updates from {len(results)} clients for round {server_round}"
        logger.info(agg_msg)

        # Call the parent class's aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert parameters to ndarrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Update the server-side model with the aggregated parameters
            set_weights(self.model, aggregated_ndarrays)

            # Save the model checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"model_round_{server_round}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)

            # Also save as the latest model
            latest_path = os.path.join(self.save_dir, f"global_model_{self.metric}.pt")
            torch.save(self.model.state_dict(), latest_path)

            # Log metrics to WandB if enabled
            if self.use_wandb and "train_loss" in aggregated_metrics:
                wandb.log({"train_loss": aggregated_metrics["train_loss"]}, step=server_round)  # type: ignore

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients.

        Args:
            server_round: Current round of federated learning
            results: List of tuples of (client, evaluation result)
            failures: List of failures that occurred during evaluation

        Returns:
            Tuple of (aggregated loss, aggregation metrics)
        """
        # Call the parent method to aggregate
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log metrics to WandB if enabled
        if self.use_wandb and aggregated_loss is not None:
            metrics_dict = {"val_loss": aggregated_loss}

            # Add any additional metrics
            if aggregated_metrics:
                for key, value in aggregated_metrics.items():
                    metrics_dict[f"val_{key}"] = value

            wandb.log(metrics_dict, step=server_round)  # type: ignore

            # Update best metrics if improved
            if aggregated_loss < self.best_metrics["loss"]:
                self.best_metrics["loss"] = aggregated_loss

                # Save best model
                best_path = os.path.join(self.save_dir, f"best_model_{self.metric}.pt")
                torch.save(self.model.state_dict(), best_path)

                best_msg = f"New best model saved with loss: {aggregated_loss:.6f}"
                logger.info(best_msg)

        return aggregated_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model parameters on the server.

        This method is called if an evaluate_fn is provided to the strategy.

        Args:
            server_round: Current round of federated learning
            parameters: Current global model parameters

        Returns:
            Tuple of (loss, metrics) or None
        """
        if self.evaluate_fn is None:
            return None

        eval_res = self.evaluate_fn(server_round, parameters, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res

        # Log centralized evaluation metrics
        if self.use_wandb:
            wandb.log(  # type: ignore
                {"centralized_loss": loss, **{f"centralized_{k}": v for k, v in metrics.items()}},
                step=server_round,
            )

        return loss, metrics


def train_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics from multiple clients.

    Args:
        metrics: List of tuples (num_examples, metrics_dict)

    Returns:
        Aggregated metrics dictionary
    """
    # Extract relevant metrics
    client_metrics = []
    for num_examples, metric_dict in metrics:
        if "train_loss" in metric_dict:
            client_metrics.append((num_examples, metric_dict["train_loss"]))

    # Compute weighted average of the loss
    total_examples = sum(num_examples for num_examples, _ in client_metrics)
    weighted_loss = sum(num_examples * loss for num_examples, loss in client_metrics)

    # Return aggregated metrics
    return {"train_loss": weighted_loss / total_examples if total_examples > 0 else 0}


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients.

    Args:
        metrics: List of tuples (num_examples, metrics_dict)

    Returns:
        Aggregated metrics dictionary
    """
    # Extract relevant metrics
    val_losses = []

    for num_examples, metric_dict in metrics:
        if "val_loss" in metric_dict:
            val_losses.append((num_examples, metric_dict["val_loss"]))

    # Compute weighted average
    total_examples = sum(num_examples for num_examples, _ in val_losses)
    weighted_loss = sum(num_examples * loss for num_examples, loss in val_losses)

    # Return aggregated metrics
    aggregated_metrics = {}

    if total_examples > 0:
        aggregated_metrics["val_loss"] = weighted_loss / total_examples

    return aggregated_metrics
