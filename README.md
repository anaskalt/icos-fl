# ICOS FL: Federated Learning for Resource Monitoring

<div align="center">

[![GitHub - Version](https://img.shields.io/github/v/release/anaskalt/icos-fl?include_prereleases)](https://github.com/anaskalt/icos-fl/releases)
[![GitHub - Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/anaskalt/icos-fl)
[![Read the Docs](https://readthedocs.org/projects/icos-fl/badge/)](https://icos-fl.readthedocs.io/)
[![License](https://img.shields.io/github/license/anaskalt/icos-fl.svg)](https://github.com/anaskalt/icos-fl/blob/master/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

</div>

ICOS-FL is a federated learning framework powered by Flower for real-time resource monitoring and prediction. It enables distributed training of LSTM models to forecast CPU usage, memory consumption, and power consumption across ICOS nodes while preserving data privacy.

## Features

- **Federated Learning**: Train models across distributed nodes while keeping data local
- **Privacy-Preserving**: Learn from system metrics without centralizing sensitive data
- **Real-time Monitoring**: Track CPU, memory, and power consumption metrics
- **LSTM Prediction**: Forecast resource usage with configurable time windows
- **DataClay Integration**: Efficient storage and retrieval of time series data
- **Docker Deployment**: Easy setup with containerized components
- **Modular Architecture**: Customize models, metrics, and federated strategies

## Installation

### From Source

```bash
git clone https://github.com/anaskalt/icos-fl.git
cd icos-fl
pip install -e .
```

## Quick Start

### Setup Infrastructure

```bash
# Start required services using Docker Compose
docker compose up -d
```

### Run Federated Learning

```bash
# Start FL
flwr run . local-deployment --stream

# Start FL with different configuration
flwr run . --run-config 'metric="power_consumption"' --stream
```

## Architecture

ICOS-FL consists of three main layers:

1. **Data Collection Layer**: Captures system metrics via Scaphandre and OpenTelemetry
2. **Storage Layer**: Persists time series data in DataClay with sliding window approach
3. **Learning Layer**: Implements federated learning with Flower's SuperLink and SuperNodes

The framework uses LSTM (Long Short-Term Memory) neural networks to predict resource usage patterns based on historical system metrics.

## Documentation

Comprehensive documentation is available at [https://icos-fl.readthedocs.io/](https://icos-fl.readthedocs.io/), including:

- [Introduction to ICOS-FL](https://icos-fl.readthedocs.io/en/latest/introduction/overview.html)
- [Architecture Overview](https://icos-fl.readthedocs.io/en/latest/explanation/architecture/overview.html)
- [Deployment Guide](https://icos-fl.readthedocs.io/en/latest/how_to/deployment/docker_setup.html)
- [API Reference](https://icos-fl.readthedocs.io/en/latest/reference/index.html)

## Contributing

Contributions are welcome! Please check out our [Contributing Guide](https://icos-fl.readthedocs.io/en/latest/contributing/index.html) for details on:

- [Code Guidelines](https://icos-fl.readthedocs.io/en/latest/contributing/code.html)
- [Documentation Standards](https://icos-fl.readthedocs.io/en/latest/contributing/documentation.html)
- [Testing Requirements](https://icos-fl.readthedocs.io/en/latest/contributing/testing.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ICOS-FL in your research, please cite:

```bibtex
@software{kaltakis2025icos,
  author = {Kaltakis, Anastasios},
  title = {ICOS-FL: Federated Learning Framework for Resource Monitoring},
  year = {2025},
  url = {https://github.com/anaskalt/icos-fl}
}
```
