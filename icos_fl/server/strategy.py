"""Strategy implementation for ICOS-FL federated learning.

This module defines custom strategies for federated learning aggregation,
including model update aggregation, metrics tracking, and checkpoint
management.
"""

import logging
import os
from logging import INFO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Metrics, Parameters, Scalar, logger, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import wandb
from icos_fl.models.lstm import LSTMModel, set_weights
from icos_fl.utils.logo import NEURAL_NET, OBJECT2, print_banner, print_completion_banner

logging.getLogger("flwr").propagate = False

PROJECT_NAME = "ICOS-FL"


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

        # Display strategy initialization banner
        print_banner(
            logo=OBJECT2,
            title="  ICOS-FL STRATEGY INITIALIZED",
            message=f"  Federated Learning strategy for metric: {metric}",
            show_version=False,
        )

        save_msg = f"Saving models to {save_dir}"
        logger.log(INFO, save_msg)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases project."""
        wandb.init(project=PROJECT_NAME, name=f"{self.metric}-ServerApp", config=self.run_config)  # type: ignore

    def log_metrics(self, server_round: int, metrics_dict: Dict[str, Scalar]) -> None:
        """Log metrics to Weights & Biases and optionally save them.

        This helper method logs metrics to Weights & Biases with proper
        formatting based on the metric type (training, evaluation, or centralized).

        Args:
            server_round: Current federated learning round
            metrics_dict: Dictionary containing metrics to log
        """
        if not self.use_wandb:
            return

        # Log metrics to WandB with appropriate step
        wandb.log(metrics_dict, step=server_round)  # type: ignore

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
        logger.log(INFO, agg_msg)

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

            # Prepare and log training metrics
            training_metrics = {}
            if "train_loss" in aggregated_metrics:
                training_metrics["federated_training_loss"] = aggregated_metrics["train_loss"]
                training_metrics["federated_training_accuracy"] = max(
                    0.0, 1.0 - training_metrics["federated_training_loss"]
                )

            # Log training metrics to WandB
            self.log_metrics(server_round, training_metrics)

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

        # Prepare evaluation metrics
        if aggregated_loss is not None:
            evaluation_metrics = {
                "federated_evaluate_loss": aggregated_loss,
                "federated_evaluate_accuracy": max(0.0, 1.0 - aggregated_loss),
            }

            # Log evaluation metrics to WandB
            self.log_metrics(server_round, evaluation_metrics)

            # Update best metrics if improved
            if aggregated_loss < self.best_metrics["loss"]:
                self.best_metrics["loss"] = aggregated_loss

                # Save best model
                best_path = os.path.join(self.save_dir, f"best_model_{self.metric}.pt")
                torch.save(self.model.state_dict(), best_path)

                # Show banner for best model
                print_banner(
                    logo=NEURAL_NET,
                    title="  BEST MODEL SAVED",
                    message=f"  New best model with loss: {aggregated_loss:.6f}",
                    show_version=False,
                )

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

        # Prepare centralized metrics
        centralized_metrics = {"centralized_loss": loss}

        # Add centralized_accuracy
        if "centralized_accuracy" in metrics:
            centralized_metrics["centralized_accuracy"] = metrics["centralized_accuracy"]
        else:
            # Create a synthetic accuracy measure if not provided
            centralized_metrics["centralized_accuracy"] = max(0.0, 1.0 - loss)

        # Log centralized metrics
        self.log_metrics(server_round, centralized_metrics)

        # Check if this is the final round
        num_rounds = self.run_config.get("num-server-rounds", 10)
        if server_round == num_rounds:
            # This is the final round, show completion banner
            print_completion_banner()

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
    client_accuracies = []

    for num_examples, metric_dict in metrics:
        if "train_loss" in metric_dict:
            client_metrics.append((num_examples, metric_dict["train_loss"]))
            # Calculate accuracy from loss
            accuracy = max(0.0, 1.0 - metric_dict["train_loss"])
            client_accuracies.append((num_examples, accuracy))

    # Compute weighted average of the loss
    total_examples = sum(num_examples for num_examples, _ in client_metrics)

    if total_examples > 0:
        weighted_loss = sum(num_examples * loss for num_examples, loss in client_metrics)
        weighted_accuracy = sum(num_examples * acc for num_examples, acc in client_accuracies)

        return {
            "train_loss": weighted_loss / total_examples,
            "train_accuracy": weighted_accuracy / total_examples,
        }
    else:
        return {"train_loss": 0.0, "train_accuracy": 0.0}


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients.

    Args:
        metrics: List of tuples (num_examples, metrics_dict)

    Returns:
        Aggregated metrics dictionary
    """
    # Extract relevant metrics
    val_losses = []
    val_accuracies = []

    for num_examples, metric_dict in metrics:
        if "val_loss" in metric_dict:
            val_losses.append((num_examples, metric_dict["val_loss"]))
            # Calculate accuracy from loss
            accuracy = max(0.0, 1.0 - metric_dict["val_loss"])
            val_accuracies.append((num_examples, accuracy))

    # Compute weighted average
    total_examples = sum(num_examples for num_examples, _ in val_losses)

    aggregated_metrics = {}

    if total_examples > 0:
        weighted_loss = sum(num_examples * loss for num_examples, loss in val_losses)
        weighted_accuracy = sum(num_examples * acc for num_examples, acc in val_accuracies)

        aggregated_metrics["val_loss"] = weighted_loss / total_examples
        aggregated_metrics["accuracy"] = weighted_accuracy / total_examples

    return aggregated_metrics
