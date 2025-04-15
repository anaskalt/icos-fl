# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-04-15

### Added
- Implement LSTM model for time series prediction
- Create Processor class for time series data preparation
- Implement TimeSeriesData and Fetcher for DataClay integration
- Add CustomFedAvg strategy with metrics tracking and model checkpointing
- Implement client with Processor integration and data fetching
- Implement server with centralized evaluation capabilities
- Apply Singleton pattern to optimize DataClay connections
- Add ASCII banner system for enhanced user experience
- Add server round tracking and standardize WandB metrics
- Refactor deployment setup for host network mode and improved architecture
- Implement robust model artifacts storage with fallback mechanism

### Fixed
- Resolve Sphinx documentation build warnings
- Add null check for evaluation result in strategy
- Remove redundant type ignore comments in logger
- Secure implementation of temporary directories for model storage

### Changed
- Standardize metric names for clarity and consistency
- Replace custom Logger with Flower logger
- Use default Logger configuration parameters
- Restructure to follow Flower App Bundle (FAB) conventions
- Remove unused 'metric' parameter from LSTMModel
- Enhance terminal display and color utilities
- Improve storage organization following Python ML project best practices

### Documentation
- Update README with comprehensive project description
- Update documentation dependencies and sphinx config
- Restructure documentation following the Diátaxis framework
- Add dataflow pipeline documentation
- Add detailed deployment architecture documentation
- Update deployment and configuration guides with correct container settings
- Refocus quickstart guide to properly focus on Flower FL setup
- Fix command line argument formats in tuning guides
- Update ReadTheDocs configuration to use latest Ubuntu and Python
- Fix documentation typos and improve navigation

### Dependencies
- Add pandas>=2.2.3 for data manipulation
- Add scikit-learn>=1.6.1 for machine learning functionality
- Add dataclay==4.0.0 for data persistence
- Update wandb dependency to v0.19.8
- Update project configuration and pre-commit hooks

### Tests
- Add comprehensive test suite for Fetcher with integration tests
- Add tests for Processor implementation
- Add tests for LSTM model implementation
- Add comprehensive Logger test suite
- Set up initial test structure and type hints support

## [0.1.0] - 2025-04-07
### Added
- First release.

[Unreleased]: https://github.com/anaskalt/icos-fl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/anaskalt/icos-fl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anaskalt/icos-fl/releases/tag/v0.1.0
