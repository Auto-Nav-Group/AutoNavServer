from drl_utils import Quaternion
import pybullet as p
import pybullet_data
import numpy as np
import os

TIME_DELTA = 0.1 # Time setup in simulation
GUI = True # GUI flag

class DRL_VENV:
    def __init__(self, environment_dimensions, map, assets_path):
        self.basis = map
        self.robot = None
        self.environment_dimensions = environment_dimensions
        self.goal_x = 1
        self.goal_y = 0
        self.obstacles = []
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(assets_path)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        planeId = p.loadURDF("floor.urdf")
        for i in range(4):
            p.loadURDF("bound"+str(i+1)+".urdf")
        for i in range(len(self.basis.obstacles)):
            self.obstacles.append(p.loadURDF("obs_"+str(i+1)+".urdf"))
    def reset(self): # Create a new environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        planeId = p.loadURDF("plane.urdf")
        # Determine new random orientation
        angle = np.random.uniform(-np.pi, np.pi) #Generate a random angle to start at
        quaternion = Quaternion.from_euler(0, 0, angle)
