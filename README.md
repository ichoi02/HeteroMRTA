# HeteroMRTA-Priority: Teaching Multi-Robot Systems to Eat The Frog
*Original codebase adapted from: [MarmotLab_HeteroMRTA](https://github.com/marmotlab/HeteroMRTA)* <br>
**Course Project:** 16-831 Introduction to Robot Learning, Fall 2025, CMU  
**Authors:** Alina Wang (MRSD), Minsik Jeon (MSR), Esme Rubinstein (RI PhD), Shunxuan Wang (MRSD), Itak Choi (MSME) 

## Overview
This repository contains the code for our project **HeteroMRTA-Priority**, which extends the state-of-the-art in learning-based Multi-Robot Task Allocation (MRTA). We build upon the work of Dai et al. to solve the Single-Task Robot, Multi-Robot Task (ST-MR-TA) allocation problem using deep reinforcement learning.

While traditional methods often struggle with scalability or fail to handle complex constraints, our approach introduces a **priority-weighted reward mechanism**. This encourages heterogeneous agents to "eat the frog"—prioritizing high-importance tasks significantly earlier in the schedule without compromising the overall makespan or system scalability.

## Key Features
* **Priority-Aware RL:** Integrates a priority term $P$ into the standard reward function ($R(\Phi) = -T - W + \omega_P P$) to model task urgency.
* **Scalability:** Demonstrated performance on large-scale scenarios with up to **150 agents and 500 tasks**, maintaining computation times within 2% of non-prioritized baselines.
* **Heterogeneous Teams:** Utilizes the species-traits model to coordinate agents with unique skill sets.
* **Decentralized Execution:** Trains agents to make sequential decisions that allow for decentralized task selection during execution.

## Demo
<img src="env/demo.gif" alt="demo" style="width: 70%;">

*Visual comparison of baseline vs. priority-based agents. Darker polygons represent higher-priority tasks.*

## Code Structure
The codebase is organized into three main components:
1.  **Environments:** Implements the species-traits model, generating random task locations with integer-based priority levels ($p_j \in [-5, 5]$) and agent depots.
2.  **Neural Network:** An attention-based actor-critic network architecture implemented in PyTorch.
3.  **Ray Framework:** Implementation of the algorithm with a POMO-inspired baseline for efficient training.

## Running Instructions
### 1. Training
To train the model, first configure the hyperparameters (including `priority_weight`) in `parameters.py`.

``` bash
python driver.py
```

### 2. Testing
To evaluate the trained model and generate performance metrics (Makespan, Priority Reward, Wait Time):

```bash
python test.py
```

### 3. Requirements
* Python >= 3.6
* Torch >= 1.8.1
* Numpy
* Ray
* Matplotlib
* Scipy
* Pandas

