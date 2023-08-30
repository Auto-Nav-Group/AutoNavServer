from drl_utils import Quaternion
import pybullet as p
import pybullet_data
import numpy as np

TIME_DELTA = 0.1 # Time setup in simulation

class DRL_VENV:
    def __init__(self, environment_dimensions, map):
        self.basis = map
        self.robot = None
        self.environment_dimensions = environment_dimensions
        self.goal_x = 1
        self.goal_y = 0
        self.obstacles = []
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        #planeId = p.loadURDF("plane.urdf")
        for i in range(len(self.basis.obstacles)):
            self.obstacles.append(p.loadURDF("cube.urdf", [self.basis.obstacles[i].Loc.x-map.size.width/2, self.basis.obstacles[i].Loc.y-map.size.height/2, 0.5], [0,0,0,1], 0))
    def reset(self): # Create a new environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        planeId = p.loadURDF("plane.urdf")
        # Determine new random orientation
        angle = np.random.uniform(-np.pi, np.pi) #Generate a random angle to start at
        quaternion = Quaternion.from_euler(0, 0, angle)
