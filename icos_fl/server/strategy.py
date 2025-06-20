"""Strategy implementation for ICOS-FL federated learning.

This module defines custom strategies for federated learning aggregation,
including model update aggregation, metrics tracking, checkpoint management,
and multi-backend model persistence.
"""

import logging
import os
from datetime import datetime
from logging import ERROR, INFO, WARN
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from flwr.common import FitRes, Metrics, Parameters, Scalar, logger, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from icos_fl.models.lstm import LSTMModel, set_weights
from icos_fl.utils.logo import NEURAL_NET, OBJECT2, print_banner, print_completion_banner
from icos_fl.utils.model_storage import ModelStorageManager

logging.getLogger("flwr").propagate = False

PROJECT_NAME = "ICOS-FL"


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with multi-backend model persistence support.

    Extends the standard FedAvg strategy with additional functionality:
    - DataClay distributed storage for trained models
    - Local filesystem storage with atomic writes
    - Weights & Biases metrics tracking
    - Comprehensive model checkpointing
    - Flexible storage backend configuration

    Storage behavior is controlled via run_config parameters:
    - save-local: Enable/disable local filesystem storage (default: True)
    - save-dataclay: Enable/disable DataClay storage (default: False)

    Args:
        *args: Variable length argument list for FedAvg parent class
        **kwargs: Arbitrary keyword arguments for FedAvg parent class
        run_config: Configuration dictionary from context
        model: The LSTM model used for federated learning
        metric: The metric being predicted (cpu_usage, memory_usage, power_consumption)
        use_wandb: Whether to use Weights & Biases for metrics logging
        save_dir: Optional directory for local model checkpoints
    """

    def __init__(
        self,
        run_config: Dict[str, Any],
        model: LSTMModel,
        metric: str,
        use_wandb: bool = True,
        save_dir: Optional[str] = None,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the custom FedAvg strategy with storage configuration.

        Storage options are read from run_config:
        - 'save-local': Whether to save models locally (default: True)
        - 'save-dataclay': Whether to save models to DataClay (default: False)
        - 'dataclay-host': DataClay proxy host (default: "127.0.0.1")
        - 'dataclay-dataset': DataClay dataset name (default: "admin")

        Key Implementation Details:
        - If neither storage backend is enabled, defaults to local storage
        - DataClay initialization failures trigger automatic fallback to local storage
        - The save_dir parameter is only required when save-local is True
        """
        super().__init__(*args, **kwargs)

        self.run_config = run_config
        self.model = model
        self.metric = metric
        self.use_wandb = use_wandb
        self.save_dir = save_dir

        # Extract storage configuration from run_config
        self.save_local = run_config.get("save-local", True)
        self.save_dataclay = run_config.get("save-dataclay", False)

        # Validate storage configuration to ensure at least one backend is operational
        if not self.save_local and not self.save_dataclay:
            if self.save_dir is not None:
                # Enable local storage when directory is available
                self.save_local = True
                logger.log(INFO, "No storage backend configured - enabling local storage")
            else:
                # Critical: No viable storage backend available
                logger.log(
                    WARN,
                    "No storage backend available and no save directory provided - "
                    "models will not be persisted",
                )

        # Model configuration for reconstruction
        self.model_config = {
            "hidden_layer_size": model.hidden_layer_size,
            "time_step": model.time_step,
            "num_layers": model.num_layers,
            "class": model.__class__.__name__,
            "module": model.__class__.__module__,
        }

        # Initialize storage manager if DataClay is enabled
        self.storage_manager: Optional[ModelStorageManager] = None
        if self.save_dataclay:
            try:
                # Get DataClay config from run_config
                dataclay_host = run_config.get("dataclay-host", "127.0.0.1")
                dataclay_dataset = run_config.get("dataclay-dataset", "admin")

                self.storage_manager = ModelStorageManager(
                    proxy_host=dataclay_host, dataset=dataclay_dataset
                )

                dataclay_info_msg = (
                    f"DataClay storage initialized - host: {dataclay_host}, "
                    f"dataset: {dataclay_dataset}"
                )
                logger.log(INFO, dataclay_info_msg)

            except (ImportError, ConnectionError, RuntimeError) as e:
                error_msg = f"Failed to initialize DataClay storage: {e}"
                logger.log(ERROR, error_msg)

                # Fallback to local storage only if save_dir is available
                self.save_dataclay = False
                if self.save_dir is not None:
                    fallback_msg = "Falling back to local storage"
                    logger.log(INFO, fallback_msg)
                    self.save_local = True
                else:
                    # Cannot fallback to local storage without directory
                    no_fallback_msg = (
                        "Cannot fallback to local storage - no save directory provided. "
                        "Models will not be persisted"
                    )
                    logger.log(WARN, no_fallback_msg)

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

    def _save_model_local(self, save_path: str, metadata: Dict[str, Any]) -> bool:
        """Save model locally with atomic writes.

        Note: Directory creation is handled by server.py during initialization.
        This method assumes the directory already exists.

        Args:
            save_path: Full path for saving the model
            metadata: Metadata to include in checkpoint

        Returns:
            True if save successful, False otherwise
        """
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "metadata": metadata,
            }

            # Atomic write to prevent corruption
            temp_path = f"{save_path}.tmp"
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, save_path)
            return True

        except (OSError, IOError) as e:
            fallback_error_msg = f"Local save failed: {type(e).__name__}: {e}"
            logger.log(ERROR, fallback_error_msg)
            return False

    def _save_model_backends(
        self, save_path: Optional[str], identifier: str, metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Save model to configured backends.

        Args:
            save_path: Path for local save (can be None if local storage is disabled)
            identifier: Identifier for DataClay save
            metadata: Metadata for the save

        Returns:
            Dictionary with save results for each backend
        """
        results = {"local": False, "dataclay": False}

        if self.storage_manager:
            results = self.storage_manager.save_model(
                model=self.model,
                save_path=save_path or "",
                identifier=identifier,
                use_local=self.save_local and save_path is not None,
                use_dataclay=self.save_dataclay,
                model_config=self.model_config,
                metadata=metadata,
            )
        elif self.save_local and save_path is not None:
            # Direct local save fallback
            results["local"] = self._save_model_local(save_path, metadata)

        return results

    def _save_model(self, model_type: str, round_number: Optional[int] = None) -> None:
        """Save model to configured storage backends with proper error handling.

        Centralizes model saving logic with consistent naming conventions
        and comprehensive metadata tracking across all model types.

        Args:
            model_type: Model variant to save ('latest', 'best', 'checkpoint')
            round_number: FL round number for checkpoint models

        Raises:
            ValueError: If model_type is not recognized
        """
        # Validate model type
        valid_types = {"latest", "best", "checkpoint"}
        if model_type not in valid_types:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_types}")

        # Determine storage paths and identifiers
        if model_type == "checkpoint":
            if round_number is None:
                raise ValueError("round_number required for checkpoint models")
            filename = f"model_round_{round_number}.pt"
            identifier = f"{self.metric}_checkpoint_r{round_number}"
        elif model_type == "latest":
            filename = f"global_model_{self.metric}.pt"
            identifier = f"{self.metric}_latest"
        else:  # model_type == "best"
            filename = f"best_model_{self.metric}.pt"
            identifier = f"{self.metric}_best"

        # Construct full save path only if save_dir is available
        save_path = None
        if self.save_dir is not None and self.save_local:
            save_path = os.path.join(self.save_dir, filename)

        # Prepare comprehensive metadata
        metadata = {
            "metric": self.metric,
            "model_type": model_type,
            "round": round_number,
            "best_loss": self.best_metrics.get("loss"),
            "timestamp": datetime.now().isoformat(),
            "run_config": {
                "hidden_layer_size": self.model_config.get("hidden_layer_size"),
                "time_step": self.model_config.get("time_step"),
                "num_layers": self.model_config.get("num_layers"),
            },
        }

        # Execute save operation
        results = self._save_model_backends(save_path, identifier, metadata)

        # Log results
        if results["local"]:
            local_save_msg = f"Model saved to local storage: {filename}"
            logger.log(INFO, local_save_msg)

        if results["dataclay"]:
            dataclay_save_msg = f"Model persisted to DataClay: {identifier}"
            logger.log(INFO, dataclay_save_msg)

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

            # Save models using centralized method
            self._save_model("checkpoint", round_number=server_round)
            self._save_model("latest", round_number=server_round)

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
                self._save_model("best")

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

            # Clean up storage manager if present
            if self.storage_manager:
                try:
                    self.storage_manager.cleanup()
                    cleanup_msg = "Storage manager resources cleaned up"
                    logger.log(INFO, cleanup_msg)
                except (AttributeError, RuntimeError) as e:
                    cleanup_error_msg = (
                        f"Error during storage manager cleanup: {type(e).__name__}: {e}"
                    )
                    logger.log(WARN, cleanup_error_msg)

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
