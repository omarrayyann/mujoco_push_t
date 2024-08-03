# PushT Task with MuJoCo
**Part of the [MujocoAR](https://github.com/omarrayyann/MujocoAR) package demos**

A MuJoCo environment (```scene.xml```) simulation of a pushT task, a common benchmark for testing behaviors multimodality. The robot's goal is to push a "T" shaped object to a target location. The simulation includes an operational space controller to handle the movement of a KUKA-iiwa14 arm with a stick tool at the end.

Main methods and attributes:

- `__init__(self)`: Initializes the simulation environment, controller, and MujocoAR connector.
- `random_placement(self, min_seperation=0.1, max_seperation=0.3)`: Places the objects randomly in the scene with specified separation constraints.
- `start(self)`: Starts the simulation and control loop.
- `done(self)`: Checks if the task is completed.
- `fell(self)`: Checks if the object has fallen.
