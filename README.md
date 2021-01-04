# Short-term bus travel time prediction for transfer synchronization with intelligent uncertainty handling

This repository consist of the source code used for producing the results in the paper:

Petersen, N. C., Parslov, A., & Rodrigues, F. (2020). *Short-term bus travel time prediction for transfer synchronization with intelligent uncertainty handling.* Submitted to Expert System with Applications.

## Abstract
This paper presents two novel approaches for uncertainty estimation adapted and extended for the multi-link bus travel time problem. The uncertainty is modeled directly as part of recurrent artificial neural networks, but using two fundamentally different approaches: one based on *Deep Quantile Regression* (DQR) and the other on *Baysian Recurrent Neural Network* (BRNN). Both models predict multiple time steps into the future but handle the time-dependent uncertainty estimation differently. We present a sampling technique in order to aggregate quantile estimates for link level travel time to yield the multi-link travel time distribution needed for a vehicle to travel from its current position to a specific downstream stop point or transfer site. 

To motivate the relevance of *uncertainty-aware* models in the domain, we focus on the connection assurance application as a case study: An expert system to determine whether a bus driver should hold and wait for a connecting service, or break the connection and reduce its own delay. Our results show that the DQR-model performs overall best for the 80%, 90% and 95% prediction intervals, both for a 15 minute time horizon into the future (t + 1), but also for the 30 and 45 minutes time horizon (t + 2 and t + 3), with a constant, but very small underestimation of the uncertainty interval (1-4 pp.). But we also show, that the BRNN model still can outperform the DQR for specific cases. Finally, we demonstrate how a simple decision support system can take advantage of our *uncertainty-aware* travel time models to prioritize the difference in travel time uncertainty for bus holding at strategic points, reducing the introduced delay for the connection assurance application.

## Data
The raw data for the 200S and 300S dataset is stored in ``data/200S.csv.gz`` and ``data/300S.csv.gz``. We also include pre-processed datasets (raw, descaled and centered) for the 15min regular time series, devided in train/validation/test in ``200S_15min_(...).csv.gz`` and ``300S_15min_(...).csv.gz`` files.

## DQR model files
After running the DQR model, result files (e.g. predicted quantiles, drawn samples, model check points) will be stored in the ``dqr`` folder.
- ``dqr-param-search-joint.py``: Performs hyper parameter tuning for DQR model.
- ``dqr-joint-200S.ipynb`` / ``dqr-joint-300S.ipynb``: Notebooks containing train/predict for the 200S and 300S dataset
- ``dqr-draw-samples-joint-200S.py`` / ``dqr-draw-samples-joint-300S.py``: Draw samples using the curve-fitting approach described in the paper for the 200S and 300S test dataset
- ``dqr-results-joint-300S.ipynb``: Creates tables with results presented in the paper

## BRNN model files
After running the BRNN model, result files (e.g. drawn samples) will be stored in the ``brnn`` folder.
- ``brnn-param-search-blitz.py`` Performs hyper parameter tuning for BRNN model.
- ``brrn-blitz-raw-200S`` / ``brnn-blitz-raw-300S``: Notebooks containing train/predict for the 200S and 300S dataset. Results are included in the bottom.

## Other files
- ``baseline-kalman.ipynb``: Baseline with linear Kalman filter.
- ``plots-*.ipynb``: Create all plots used in the paper.
