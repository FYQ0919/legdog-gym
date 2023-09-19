# Capdog
Capability planner and controller for hexapod robot -- Littledog

## rl support
Reinforcement learning simulation environment for legged robots

## install
pip install -r requirement.txt

## use
1.  Robot model is in /model, where you can change Littledog's mujoco xml, including actuators and sensors.
2.  Robot perception and control is in dog_*.py files, modify them to implement your own control code or perception code.
3.  Reinforcement learning code for Littledog is in /learning. You can implement your own networks and agent here.

## simulation & debugging
When you write your own control code in dog_*.py files. You can test how it works by running dog_model.py. Simulation scence is as the following picture:
![](https://github.com/lonelyfluency/Capdog/blob/main/figs/littledog.png)
