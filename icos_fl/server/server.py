"""ICOS-FL Server Implementation.

This module implements the server-side components of the ICOS Federated Learning system.
It provides the core server functionality for coordinating federated learning across 
multiple clients, managing the global model, and handling the federation strategy.

The server is responsible for:
    - Global model initialization and management
    - Client coordination and synchronization
    - Model update aggregation
    - Training progress monitoring
    - Model persistence and checkpointing
    - Federation strategy configuration

Key Components:
    - FLServer: Main server class implementing federated learning logic
    - Configuration filters for model, federation, and logging settings
    - Custom federation strategy based on FedAvg
    
Example:
    >>> from icos_fl.server import FLServer
    >>> config = ConfigFetcher("server.yaml").get_configuration()
    >>> server = FLServer(config)
    >>> server.run()

Note:
    This implementation uses the Flower (flwr) framework for federated learning
    primitives and builds upon it with custom aggregation and coordination logic.
"""

import numpy as np
import flwr as fl
from typing import Dict, List, Tuple
from pathlib import Path
from icos_fl.models.lstm import LSTMModel
from icos_fl.utils.logger import ICOSLogger as LOGGER

LOG = LOGGER("Server")

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
        "model-learning-rate": "learning_rate",
        "model-checkpoint-path": "checkpoint_path",
        "model-checkpoint-interval": "checkpoint_interval"
    }
    
    return {model_key_map[k]: v for k, v in confdata.items() 
            if k in model_key_map}

def filter_federation_args(confdata: Dict) -> Dict:
    """Extract and validate federation strategy configuration.
    
    Args:
        confdata: Raw configuration dictionary
        
    Returns:
        Dict containing federation parameters with standardized keys
    """
    federation_key_map = {
        "federation-address": "address",
        "federation-port": "port",
        "federation-min-clients": "min_clients",
        "federation-rounds": "rounds",
        "federation-aggregation-timeout": "timeout"
    }
    
    return {federation_key_map[k]: v for k, v in confdata.items()
                if k in federation_key_map}
    
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

class FLServer:
    """ICOS Federated Learning Server implementation.
    
    This class implements the server-side logic of the federated learning system.
    It manages the global model state, coordinates training across clients, and 
    handles model aggregation and persistence.
    
    Key Features:
        - Configurable federation strategy
        - Client coordination and synchronization
        - Model state management and persistence
        - Metrics aggregation and monitoring
        - Checkpoint management
    
    Attributes:
        model_config (Dict): LSTM model configuration 
        federation_config (Dict): Federation settings
        model (LSTMModel): Global LSTM model instance
        strategy (fl.server.strategy.Strategy): Federation strategy
    """

    def __init__(self, confdata: Dict):
        """Initialize server components.
        
        Args:
            confdata: Parsed configuration dictionary
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Filter configurations
            self.model_config = filter_model_args(confdata)
            self.federation_config = filter_federation_args(confdata)
            
            # Configure logger
            LOG.reconfigure(filter_logger_args(confdata))
            
            # Initialize components
            self.model = self._init_global_model()
            self.strategy = self._create_strategy()
            
            self._validate_configuration()
            LOG.info("Server initialized successfully")
            
        except Exception as e:
            LOG.error(f"Server initialization failed: {str(e)}")
            raise RuntimeError(f"Server initialization failed: {str(e)}")

    def _validate_configuration(self) -> None:
        """Validate server configuration.
        
        Performs validation checks on model and federation configurations
        to ensure all required parameters are present and valid.
        
        Raises:
            ValueError: If configuration validation fails
        """
        required_model = ['input_size', 'hidden_size', 'num_layers', 'output_size']
        required_fed = ['address', 'port', 'min_clients', 'rounds']
        
        missing_model = [k for k in required_model if k not in self.model_config]
        missing_fed = [k for k in required_fed if k not in self.federation_config]
        
        if missing_model or missing_fed:
            raise ValueError(
                f"Missing configuration parameters: "
                f"model={missing_model}, federation={missing_fed}"
            )

    def _init_global_model(self) -> LSTMModel:
        """Initialize global LSTM model.
        
        Creates and configures the global model instance that will be shared
        with clients and used for aggregation.
        
        Returns:
            Configured LSTM model instance
            
        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            model = LSTMModel(**self.model_config)
            LOG.info("Global model initialized with config:", 
                     extra={"config": self.model_config})
            return model
        except Exception as e:
            LOG.error(f"Failed to initialize global model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _create_strategy(self) -> fl.server.strategy.Strategy:
        """Configure federated learning strategy.
        
        Creates and configures the federation strategy that handles
        client selection, model aggregation, and round management.
        
        Returns:
            Configured federation strategy instance
            
        Raises:
            RuntimeError: If strategy creation fails
        """
        try:
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=self.federation_config['min_clients'],
                min_evaluate_clients=self.federation_config['min_clients'], 
                min_available_clients=self.federation_config['min_clients'],
                on_fit_config_fn=self._get_fit_config,
                evaluate_metrics_aggregation_fn=self._aggregate_metrics,
                fit_metrics_aggregation_fn=self._aggregate_metrics,
                initial_parameters=fl.common.weights_to_parameters(
                    self.get_parameters()
                )
            )
            LOG.info("Federation strategy created")
            return strategy
        except Exception as e:
            LOG.error(f"Failed to create strategy: {str(e)}")
            raise RuntimeError(f"Strategy creation failed: {str(e)}")

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters.
        
        Extracts the current model parameters for distribution to clients.
        
        Returns:
            List of model parameter arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def _get_fit_config(self, server_round: int) -> Dict:
        """Get training configuration for current round.
        
        Args:
            server_round: Current federation round number
            
        Returns:
            Configuration dictionary for client training
        """
        return {
            "server_round": server_round
        }

    def _aggregate_metrics(self, metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate client metrics.
        
        Combines training metrics from multiple clients into aggregate values.
        
        Args:
            metrics: List of (num_examples, metrics_dict) tuples
            
        Returns:
            Dictionary containing aggregated metrics
        """
        if not metrics:
            return {}
            
        total_examples = sum(num_examples for num_examples, _ in metrics)
        if total_examples == 0:
            return {}
            
        total_loss = sum(num_examples * m['loss'] for num_examples, m in metrics)
        aggregated = {"loss": total_loss / total_examples}
        
        LOG.info(f"Aggregated metrics: {aggregated}")
        return aggregated

    def save_checkpoint(self, round: int) -> None:
        """Save model checkpoint.
        
        Args:
            round: Current federation round number
            
        Raises:
            RuntimeError: If checkpoint save fails
        """
        try:
            checkpoint_dir = Path(self.model_config['checkpoint_path'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            path = checkpoint_dir / f"model_round_{round}.pt"
            self.model.save(str(path))
            
            LOG.info(f"Checkpoint saved: {path}")
        except Exception as e:
            LOG.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint save failed: {str(e)}")

    def run(self) -> None:
        """Start federated learning server.
        
        Initializes and starts the Flower server with configured strategy.
        
        Raises:
            RuntimeError: If server fails to start/run
        """
        try:
            address = f"{self.federation_config['address']}:{self.federation_config['port']}"
            LOG.info(f"Starting server on {address}")
            
            fl.server.start_server(
                server_address=address,
                config=fl.server.ServerConfig(
                    num_rounds=self.federation_config['rounds']
                ),
                strategy=self.strategy
            )
        except Exception as e:
            LOG.error(f"Server failed: {str(e)}")
            raise RuntimeError(f"Server failed: {str(e)}")

def main():
    """Server entry point."""
    from icos_fl.utils.configfetcher import ConfigFetcher
    from icos_fl.utils.argparser import ICOSArgParser
    
    parser = ICOSArgParser("ICOS Federated Learning Server", "server")
    args = parser.parse()
    
    try:
        # Load and validate configuration
        config = ConfigFetcher(args.config).get_configuration()
        
        if args.dry_run:
            LOG.info("Configuration validation successful")
            return
            
        # Initialize and start server
        server = FLServer(config)
        server.run()
        
    except Exception as e:
        LOG.critical(f"Server failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()