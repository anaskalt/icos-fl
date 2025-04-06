======================
ICOS-FL Documentation
======================

.. image:: _static/images/architecture_overview.png
   :align: center
   :alt: ICOS-FL Architecture Overview

ICOS-FL is a federated learning framework for real-time resource monitoring built on Flower. It enables distributed training of LSTM models for predicting system metrics like CPU usage, memory consumption, and power usage across ICOS nodes.

.. sidebar-links::
   :home:
   :pypi:

.. toctree::
   :maxdepth: 1
   :hidden:

   Introduction <introduction/index>
   How-To Guides <how_to/index>
   Explanation <explanation/index>
   Reference <reference/index>
   Contributing <contributing/index>
   security

Features
========

* **Federated Learning**: Train models across distributed nodes while keeping data local
* **Real-time Monitoring**: Track CPU, memory, and power consumption metrics
* **LSTM Prediction**: Forecast resource usage with configurable time windows
* **DataClay Integration**: Efficient storage and retrieval of time series data
* **Docker Deployment**: Easy setup with containerized components

Quick Links
===========

* :doc:`Introduction to ICOS-FL <introduction/overview>`
* :doc:`Quick Start Guide <introduction/quickstart>`
* :doc:`Architecture Overview <explanation/architecture/overview>`
* :doc:`Deployment Guide <how_to/deployment/docker_setup>`
