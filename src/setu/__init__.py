"""SETU - source-attributed, evidence-graded trigger for urban air quality.

The package is layered so that each stage can be tested against a published
number without running the stages around it:

    io          upstream fetchers (no API keys) and the on-disk cache
    grid        node placement and the wind-aware adjacency graph
    twin        digital twin used to exercise the sensor-side algorithms
    calibrate   hygroscopic correction, regression, transfer, drift quarantine
    attribute   size-fraction source posterior with carried uncertainty
    forecast    graph nowcast and short-horizon forecast
    respond     graded local response engine
    metrics     error and threshold-classification statistics
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
