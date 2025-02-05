"""ICOS-FL Client Implementation.

This module implements the client-side components of the ICOS Federated Learning system.
It provides functionality for:
    - Local model training
    - Data fetching and preprocessing
    - Server communication
    - Model parameter synchronization
    - Training metrics collection

Key Components:
    - FLClient: Main client class implementing local training logic
    - Configuration filters for model, data, and training settings
    - Data pipeline with fetching and preprocessing
    - Local model management

The client works in conjunction with the ICOS-FL server to perform federated
learning, handling local data processing and model training while maintaining
data privacy.

Example:
    >>> from icos_fl.client import FLClient
    >>> config = ConfigFetcher("client.yaml").get_configuration()
    >>> client = FLClient(config)
    >>> client.start()

Note:
    This implementation uses the Flower (flwr) framework for federated learning
    primitives and builds upon it with custom training and data handling logic.
"""

import numpy as np
import torch
import flwr as fl
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from icos_fl.data.fetcher import DataFetcher 
from icos_fl.data.processor import DataProcessor
from icos_fl.models.lstm import LSTMModel
from icos_fl.utils.logger import ICOSLogger as LOGGER

LOG = LOGGER("Client")

def filter_server_args(confdata: Dict) -> Dict:
    """Extract and validate server configuration parameters.
    
    Args:
        confdata: Raw configuration dictionary containing all settings
        
    Returns:
        Dict containing server parameters with standardized keys
        
    Example:
        >>> conf = {'server-address': 'localhost', 'server-port': 8080}
        >>> server_args = filter_server_args(conf)
        >>> server_args
        {'address': 'localhost', 'port': 8080}
    """
    server_key_map = {
        "server-address": "address",
        "server-port": "port"
    }
    
    return {server_key_map[k]: v for k, v in confdata.items() 
            if k in server_key_map}

def filter_data_args(confdata: Dict) -> Dict:
    """Extract and validate data pipeline configuration.
    
    Args:
        confdata: Raw configuration dictionary
        
    Returns:
        Dict containing data pipeline parameters with standardized keys
        
    Example:
        >>> conf = {'data-host': 'localhost', 'data-dataset': 'metrics'}
        >>> data_args = filter_data_args(conf)
        >>> data_args
        {'host': 'localhost', 'dataset': 'metrics'}
    """
    data_key_map = {
        "data-host": "host",
        "data-dataset": "dataset",
        "data-alias": "alias",
        "data-poll-interval": "poll_interval",
        "data-timeout": "timeout",
        "data-target-metric": "target_metric"
    }
    
    filtered = {data_key_map[k]: v for k, v in confdata.items() 
               if k in data_key_map}
    
    # Set defaults
    defaults = {
        "poll_interval": 60,
        "timeout": 300
    }
    
    return {**defaults, **filtered}

def filter_training_args(confdata: Dict) -> Dict:
    """Extract and validate training configuration.
    
    Args:
        confdata: Raw configuration dictionary
        
    Returns:
        Dict containing training parameters with standardized keys
    """
    training_key_map = {
        "training-epochs": "epochs",
        "training-batch-size": "batch_size", 
        "training-time-step": "time_step",
        "training-fill-nan": "fill_nan",
        "training-train-test-split": "train_test_split",
        "training-learning-rate": "learning_rate"
    }
    
    filtered = {training_key_map[k]: v for k, v in confdata.items()
                if k in training_key_map}
    
    # Set defaults
    defaults = {
        "epochs": 1,
        "batch_size": 32,
        "fill_nan": 10,
        "train_test_split": 0.8,
        "learning_rate": 0.001
    }
    
    return {**defaults, **filtered}

def filter_model_args(confdata: Dict) -> Dict:
    """Extract and validate model configuration parameters.
    
    Args:
        confdata: Raw configuration dictionary containing all settings
        
    Returns:
        Dict containing model parameters with standardized keys
        
    Example:
        >>> conf = {'model-input-size': 10, 'model-hidden-size': 32}
        >>> model_args = filter_model_args(conf)
        >>> model_args
        {'input_size': 10, 'hidden_size': 32}
    """
    model_key_map = {
        "model-input-size": "input_size",
        "model-hidden-size": "hidden_size", 
        "model-num-layers": "num_layers",
        "model-output-size": "output_size",
        "model-learning-rate": "learning_rate"
    }
    
    return {model_key_map[k]: v for k, v in confdata.items() 
            if k in model_key_map}

def filter_logger_args(confdata: Dict) -> Dict:
    """Extract and validate logger configuration.
    
    Args:
        confdata: Raw configuration dictionary
        
    Returns:
        Dict containing logger parameters with standardized keys
    """
    logger_key_map = {
        "log-level": "level",
        "log-file": "logfile",
        "log-max-bytes": "maxbytes",
        "log-backup-count": "backupcount",
        "log-use-console": "useconsole", 
        "log-use-color": "usecolor"
    }
    
    return {logger_key_map[k]: v for k, v in confdata.items()
            if k in logger_key_map}

class FLClient(fl.client.NumPyClient):
    """ICOS Federated Learning Client implementation.
    
    This class handles the client-side operations of federated learning:
        - Data fetching and preprocessing
        - Local model training
        - Model parameter synchronization
        - Server communication
        - Training metrics collection
    
    The client maintains a local model that is synchronized with the global
    model on the server, while keeping training data private.
    
    Attributes:
        model (LSTMModel): Local model instance
        data_fetcher (DataFetcher): Data pipeline component
        data_processor (DataProcessor): Data preprocessing component
        model_config (Dict): Model configuration
        data_config (Dict): Data pipeline settings
        training_config (Dict): Training parameters
    """

    def __init__(self, confdata: Dict):
        """Initialize client components.
        
        Args:
            confdata: Parsed configuration dictionary
            
        Raises:
            RuntimeError: If initialization fails
            ConnectionError: If data backend connection fails
            ValueError: If configuration is invalid
        """
        try:
            # Filter configurations
            self.model_config = filter_model_args(confdata)
            self.data_config = filter_data_args(confdata)
            self.training_config = filter_training_args(confdata)
            
            # Configure logger
            LOG.reconfigure(filter_logger_args(confdata))
            
            # Initialize components
            self.model = LSTMModel(**self.model_config)
            self.data_fetcher = DataFetcher(**self.data_config)
            self.data_processor = DataProcessor(**self.training_config)
            
            # Validate initialization
            self._validate_configuration()
            if not self._initialize_data_pipeline():
                raise ConnectionError("Failed to initialize data pipeline")
                
            LOG.info("Client initialized successfully")
            
        except Exception as e:
            LOG.error(f"Client initialization failed: {str(e)}")
            raise RuntimeError(f"Client initialization failed: {str(e)}")

    def _validate_configuration(self) -> None:
        """Validate client configuration.
        
        Checks all configuration parameters for completeness and validity.
        
        Raises:
            ValueError: If configuration validation fails
        """
        required_model = ['input_size', 'hidden_size', 'num_layers', 'output_size']
        required_data = ['host', 'dataset', 'alias', 'target_metric']
        
        missing_model = [k for k in required_model if k not in self.model_config]
        missing_data = [k for k in required_data if k not in self.data_config]
        
        if missing_model or missing_data:
            raise ValueError(
                f"Missing configuration parameters: "
                f"model={missing_model}, data={missing_data}"
            )

    def _initialize_data_pipeline(self) -> bool:
        """Initialize data fetching pipeline.
        
        Establishes connection to data backend and initializes time series.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.data_fetcher.connect():
                LOG.error("Failed to connect to data backend")
                return False
                
            if not self.data_fetcher.initialize_timeseries():
                LOG.error("Failed to initialize time series")
                return False
                
            return True
            
        except Exception as e:
            LOG.error(f"Data pipeline initialization failed: {str(e)}")
            return False

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get current model parameters.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of model parameter arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model with received parameters.
        
        Args:
            parameters: New model parameters from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        LOG.debug("Updated local model parameters")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data.
        
        This method:
            1. Updates local model with server parameters
            2. Fetches and preprocesses local training data
            3. Trains model for configured number of epochs
            4. Returns updated parameters and metrics
        
        Args:
            parameters: Initial model parameters
            config: Training configuration from server
            
        Returns:
            Tuple containing:
                - Updated model parameters
                - Number of training examples used
                - Training metrics dictionary
                
        Raises:
            RuntimeError: If training fails
        """
        try:
            # Update local model
            self.set_parameters(parameters)
            
            # Fetch and process data
            raw_data = self.data_fetcher.fetch_metrics(
                timeout=self.data_config['timeout']
            )
            if raw_data is None:
                raise RuntimeError("Failed to fetch training data")
                
            processed_data = self.data_processor.preprocess(raw_data)
            train_data, _ = self.data_processor.create_dataset(
                processed_data,
                metric=self.data_config['target_metric']
            )
            
            # Configure training
            train_loader = DataLoader(
                train_data,
                batch_size=self.training_config['batch_size'],
                shuffle=True
            )
            
            # Train model
            train_loss = self.model.train_epoch(train_loader)
            LOG.info(f"Training completed - Loss: {train_loss:.4f}")
            
            return self.get_parameters(config), len(train_data), {"loss": train_loss}
            
        except Exception as e:
            LOG.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration from server
            
        Returns:
            Tuple containing:
                - Loss value
                - Number of test examples
                - Evaluation metrics dictionary
        """
        try:
            self.set_parameters(parameters)
            
            # Fetch test data
            raw_data = self.data_fetcher.fetch_metrics(
                timeout=self.data_config['timeout']
            )
            if raw_data is None:
                raise RuntimeError("Failed to fetch test data")
            
            # Process and create test dataset
            processed_data = self.data_processor.preprocess(raw_data)
            _, test_data = self.data_processor.create_dataset(
                processed_data,
                metric=self.data_config['target_metric']
            )
            
            # Evaluate
            test_loader = DataLoader(
                test_data,
                batch_size=self.training_config['batch_size']
            )
            loss, accuracy = self.model.test(test_loader)
            
            metrics = {
                "loss": loss,
                "accuracy": accuracy
            }
            LOG.info(f"Evaluation completed - Metrics: {metrics}")
            
            return loss, len(test_data), metrics
            
        except Exception as e:
            LOG.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")

def main():
    """Client entry point."""
    from icos_fl.utils.configfetcher import ConfigFetcher
    from icos_fl.utils.argparser import ICOSArgParser
    
    parser = ICOSArgParser("ICOS Federated Learning Client", "client")
    args = parser.parse()
    
    try:
        # Load and validate configuration
        config = ConfigFetcher(args.config).get_configuration()
        # Filter server configuration
        server_config = filter_server_args(config)

        if args.dry_run:
            LOG.info("Configuration validation successful")
            return
            
        # Initialize and start client
        client = FLClient(config)
        fl.client.start_client(
            server_address=f"{server_config['address']}:{server_config['port']}",
            client=client
        )
        
    except Exception as e:
        LOG.critical(f"Client failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()