# Sensor Fusion and Kalman Filtering Simulation

## Overview
This project demonstrates how multiple sensors can be combined to produce a more reliable output, even when some sensors generate faulty or noisy data. A Kalman filter is applied to further improve the accuracy of the combined output.

The system is tested under different conditions such as normal noise, sudden spikes, gradual drift, and mixed faults. A simple drone navigation example is also included to show how filtering improves position tracking.

## Features
* Simulation of multiple sensors
* Fault detection (range check, spikes, stuck values)
* Weighted sensor fusion based on sensor health
* Adaptive Kalman filter for smoothing data
* Drone movement simulation with noisy measurements
* Performance evaluation using RMSE

## Scenarios Tested
* Normal
* Spike
* Drift
* Mixed

## Output
For each scenario, the following are plotted:
* Ground truth
* Raw sensor data
* Fused output
* Kalman filtered output

The drone simulation includes:
* True path
* Noisy path
* Filtered path

## Conclusion
The results show that combining sensor data improves overall reliability, and applying a Kalman filter further reduces noise and estimation error.
