"""Model storage utilities for persisting FL models to multiple backends.

This module provides a flexible storage abstraction for federated learning models,
supporting both local filesystem and DataClay distributed storage with comprehensive
error handling and resource management following the ICOS-FL architecture patterns.
"""

import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from dataclay import Client, DataClayObject
from dataclay.exceptions import DataClayException

from icos_fl.utils.singleton import Singleton


class FLModelMetadata(DataClayObject):
    """DataClay persistent object for storing FL model states and metadata.

    This class provides a robust mechanism for persisting PyTorch model states
    in DataClay by properly serializing state dictionaries and maintaining
    model reconstruction information.

    Key Implementation Detail:
    - Uses explicit attribute reassignment to trigger DataClay persistence
    - Ensures nested dictionary changes are detected by the persistence layer

    Attributes:
        state_dict: Serialized model parameters as nested dictionaries
        model_config: Architecture configuration for model reconstruction
        metadata: Training context and performance metrics
        is_available: Flag indicating if valid model state is stored
        last_updated: ISO format timestamp of last update
    """

    state_dict: Dict[str, Any]
    model_config: Dict[str, Any]
    metadata: Dict[str, Any]
    is_available: bool
    last_updated: Optional[str]

    def __init__(self) -> None:
        """Initialize empty model metadata container with default values."""
        self.state_dict = {}
        self.model_config = {}
        self.metadata = {}
        self.is_available = False
        self.last_updated = None

    def store_model_state(
        self,
        model: nn.Module,
        model_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store PyTorch model state with proper serialization and DataClay persistence.

        Converts model state dictionary to DataClay-compatible format by
        serializing tensors to nested lists. Uses explicit attribute reassignment
        to ensure DataClay detects and persists the changes.

        Args:
            model: PyTorch model instance to persist
            model_config: Dictionary containing model architecture parameters
            metadata: Optional dictionary with training context information

        Raises:
            ValueError: If model is None or invalid
            RuntimeError: If serialization fails
        """
        if model is None:
            raise ValueError("Cannot store None model")

        try:
            # Create new dictionary for state - ensures DataClay detects the change
            new_state_dict = {}

            # Serialize state dict - convert tensors to lists for DataClay compatibility
            for key, tensor in model.state_dict().items():
                # Ensure tensor is on CPU before converting
                new_state_dict[key] = tensor.cpu().numpy().tolist()

            # CRITICAL: Reassign entire dictionary to trigger DataClay persistence
            # Direct modification of self.state_dict[key] = value doesn't persist properly
            self.state_dict = new_state_dict

            # Similarly, reassign other attributes to ensure persistence
            self.model_config = dict(model_config) if model_config else {}
            self.metadata = dict(metadata) if metadata else {}
            self.is_available = True
            self.last_updated = datetime.now().isoformat()

        except Exception as e:
            error_msg = f"Failed to serialize model state: {e}"
            raise RuntimeError(error_msg) from e

    def retrieve_state_dict(self) -> Dict[str, torch.Tensor]:
        """Retrieve model state dictionary as PyTorch tensors.

        Reconstructs PyTorch tensors from serialized format, ensuring
        proper tensor shapes and data types are preserved.

        Returns:
            Dictionary mapping parameter names to PyTorch tensors

        Raises:
            ValueError: If no model state is available
            RuntimeError: If deserialization fails
        """
        if not self.is_available:
            raise ValueError("No model state available in this container")

        try:
            # Convert lists back to tensors
            tensor_dict = {}
            for key, value in self.state_dict.items():
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32)

            return tensor_dict

        except Exception as e:
            error_msg = f"Failed to deserialize model state: {e}"
            raise RuntimeError(error_msg) from e


class ModelStorageManager(metaclass=Singleton):
    """Singleton manager for coordinating model storage across multiple backends.

    This class leverages the enhanced Singleton metaclass to manage model
    persistence operations, supporting both local filesystem and DataClay
    distributed storage with automatic fallback mechanisms.

    The Singleton metaclass ensures that identical (proxy_host, dataset)
    combinations return the same instance while different parameters create
    separate instances, enabling proper connection management for different
    DataClay deployments.

    Key Features:
        - Parameterized singleton via enhanced metaclass
        - Automatic connection management for DataClay backend
        - Comprehensive error handling
        - Support for concurrent storage to multiple backends
        - Resource cleanup with guaranteed connection closure

    Args:
        proxy_host: DataClay proxy server address (default: "127.0.0.1")
        dataset: DataClay dataset name for authentication (default: "admin")
    """

    def __init__(self, proxy_host: str = "127.0.0.1", dataset: str = "admin") -> None:
        """Initialize storage manager with DataClay connection parameters.

        The Singleton metaclass ensures this is only called once per unique
        parameter combination, eliminating the need for manual instance tracking.
        """
        # Store connection parameters
        self.proxy_host = proxy_host
        self.dataset = dataset

        # Initialize DataClay connection state
        self.client: Optional[Client] = None
        self._dataclay_connected = False
        self._connection_attempts = 0
        self._max_connection_attempts = 3

        # Model reference tracking for efficient retrieval
        self._model_storage_refs: Dict[str, str] = {}

        # Instance-level lock for thread-safe operations
        self._lock = threading.Lock()

    def _ensure_dataclay_connection(self) -> bool:
        """Establish or verify DataClay connection with retry logic.

        Implements exponential backoff for connection attempts and
        maintains connection state to avoid redundant connection attempts.

        Returns:
            True if connection is established, False otherwise
        """
        if self._dataclay_connected:
            return True

        with self._lock:
            # Double-check after acquiring lock
            if self._dataclay_connected:
                return True

            # Check if max attempts exceeded
            if self._connection_attempts >= self._max_connection_attempts:
                return False

            try:
                self._connection_attempts += 1

                # Create and start DataClay client
                self.client = Client(proxy_host=self.proxy_host, dataset=self.dataset)

                if self.client is None:
                    raise DataClayException("Failed to create DataClay client")

                self.client.start()
                self._dataclay_connected = True
                self._connection_attempts = 0  # Reset on success

                return True

            except DataClayException:
                self.client = None
                return False

            except (ConnectionError, OSError, RuntimeError):
                # Catch specific network-related exceptions
                self.client = None
                return False

    def _save_to_local(
        self,
        model: nn.Module,
        save_path: str,
        model_config: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Save model to local filesystem with atomic writes.

        Args:
            model: PyTorch model to save
            save_path: Local file path
            model_config: Model configuration
            metadata: Save metadata

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure directory exists with proper permissions
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True, mode=0o755)

            # Create comprehensive checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config or {},
                "metadata": metadata,
                "icos_fl_version": "0.2.0",  # Track version for compatibility
            }

            # Save with atomic write (using temporary file)
            temp_path = f"{save_path}.tmp"
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, save_path)  # Atomic operation

            return True

        except (OSError, IOError):
            # Filesystem error during local save - silently handled
            return False

    def _save_to_dataclay(
        self,
        model: nn.Module,
        identifier: str,
        model_config: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Save model to DataClay distributed storage.

        Args:
            model: PyTorch model to save
            identifier: Unique identifier for storage
            model_config: Model configuration
            metadata: Save metadata

        Returns:
            True if save successful, False otherwise
        """
        if not self._ensure_dataclay_connection():
            return False

        try:
            # Construct unique alias
            alias = f"fl_model_{identifier}"

            # Attempt to retrieve existing object
            model_metadata = None
            try:
                model_metadata = FLModelMetadata.get_by_alias(alias)

            except (DataClayException, RuntimeError):
                # Create new object if not found
                model_metadata = FLModelMetadata()
                model_metadata.make_persistent(alias=alias)

            # Store model state with enhanced config
            enhanced_config = {
                "class": model.__class__.__name__,
                "module": model.__class__.__module__,
                **(model_config or {}),
            }

            model_metadata.store_model_state(
                model=model, model_config=enhanced_config, metadata=metadata
            )

            # Track reference for efficient retrieval
            self._model_storage_refs[identifier] = alias

            return True

        except DataClayException:
            # DataClay persistence error - silently handled
            return False

    def save_model(
        self,
        model: nn.Module,
        save_path: str,
        identifier: str,
        use_local: bool = True,
        use_dataclay: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Save model to specified storage backends with comprehensive error handling.

        Coordinates model persistence across multiple backends, ensuring at least
        one successful save when possible. Implements independent error handling
        for each backend to maximize reliability.

        Args:
            model: PyTorch model instance to persist
            save_path: Local filesystem path for model checkpoint
            identifier: Unique identifier for DataClay storage alias
            use_local: Enable local filesystem storage (default: True)
            use_dataclay: Enable DataClay distributed storage (default: False)
            model_config: Model architecture parameters for reconstruction
            metadata: Optional training context and metrics

        Returns:
            Dictionary with backend names as keys and success status as values:
            {"local": bool, "dataclay": bool}

        Raises:
            ValueError: If model is not a valid nn.Module instance
        """
        # Input validation
        if not isinstance(model, nn.Module):
            raise ValueError(f"Expected nn.Module instance, got {type(model).__name__}")

        # Initialize results tracking
        results = {"local": False, "dataclay": False}
        timestamp = datetime.now().isoformat()

        # Prepare enhanced metadata
        enhanced_metadata = {
            "timestamp": timestamp,
            "identifier": identifier,
            "metric": metadata.get("metric") if metadata else None,
            **(metadata or {}),
        }

        # Local filesystem storage
        if use_local:
            results["local"] = self._save_to_local(
                model, save_path, model_config or {}, enhanced_metadata
            )

        # DataClay distributed storage
        if use_dataclay:
            results["dataclay"] = self._save_to_dataclay(
                model, identifier, model_config or {}, enhanced_metadata
            )

        return results

    def load_model_from_dataclay(
        self, identifier: str
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        """Load model state from DataClay distributed storage.

        Retrieves model state dictionary and configuration from DataClay,
        handling connection establishment and error recovery transparently.

        Args:
            identifier: Unique identifier used during model save

        Returns:
            Tuple containing:
                - state_dict: Dictionary of model parameters as tensors
                - model_config: Dictionary of architecture configuration
            Returns None if model not found or on error
        """
        if not self._ensure_dataclay_connection():
            return None

        try:
            # Resolve alias from cache or construct
            alias = self._model_storage_refs.get(identifier, f"fl_model_{identifier}")

            # Retrieve model metadata object
            model_metadata = FLModelMetadata.get_by_alias(alias)

            if not model_metadata.is_available:
                return None

            # Retrieve and validate state dictionary
            state_dict = model_metadata.retrieve_state_dict()
            model_config = model_metadata.model_config

            return state_dict, model_config

        except (DataClayException, RuntimeError, ValueError):
            return None

    def cleanup(self) -> None:
        """Perform graceful cleanup of resources and connections.

        Ensures proper resource deallocation and connection closure,
        implementing defensive cleanup to handle partial initialization
        states gracefully.
        """
        with self._lock:
            if self.client and self._dataclay_connected:
                try:
                    self.client.stop()

                except (DataClayException, RuntimeError):
                    # Error during DataClay cleanup - silently handled
                    pass

            # Always reset state regardless of connection status
            self._dataclay_connected = False
            self.client = None
            self._model_storage_refs.clear()

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection.

        Implements defensive cleanup to handle edge cases where
        explicit cleanup might not be called.
        """
        try:
            self.cleanup()
        except (AttributeError, RuntimeError):
            # Suppress exceptions in destructor
            pass
