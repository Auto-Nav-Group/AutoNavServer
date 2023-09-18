import torch

from drl_utils import Quaternion, ReplayMemory, ObjectState
import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import time
import math

TIME_DELTA = 0.1 # Time setup in simulation
GUI = False # GUI flag
GOAL_REACHED_DIST = 1 # Distance to goal to be considered reached
MIN_START_DIST = 10 # Minimum distance from start to goal
MAX_SPEED = 5 # Maximum speed of the robot
MAX_ANGULAR_SPEED = math.pi # Maximum angular speed of the robot
TIP_ANGLE = 30

LIDAR_RANGE = 10
LIDAR_ANGLE = np.pi
LIDAR_POINTS = 100

GRAVITY = 0

SPAWN_BORDER = 2

class DRL_VENV:
    def __init__(self, map, assets_path):
        self.basis = map
        self.robot = None
        self.goal = None
        self.ray_debug_id = []
        self.goal_x = 1
        self.goal_y = 0
        self.x = 0
        self.y = 0
        self.gangle = 0
        self.obstacles = []
        self.lidar_dists = []
        self.pybullet_instance = p
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(assets_path)
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        self.environment_ids = []
        self.environment_dim = 0

        self.initial_state = ObjectState()
        self.initial_state.position = [0, 0, 0.1]
        self.initial_state.orientation = [0, 0, 0, 1]

        self.floor = self.environment_ids.append(p.loadURDF("floor.urdf"))
        for i in range(4):
            self.environment_ids.append(p.loadURDF("bound"+str(i+1)+".urdf"))
        for i in range(len(self.basis.obstacles)):
            self.obstacles.append(p.loadURDF("obs_"+str(i+1)+".urdf"))
            self.environment_ids.append(self.obstacles[i])
        for i in range(len(self.environment_ids)):
            self.environment_dim += p.getNumJoints(self.environment_ids[i])
        print(self.environment_dim)
        self.robotid = p.loadURDF("robot.urdf", [0, 0, 0.1])

    def new_goal(self):
        self.goal_x = np.random.uniform(SPAWN_BORDER, self.basis.size.width-SPAWN_BORDER)
        self.goal_y = np.random.uniform(SPAWN_BORDER, self.basis.size.height-SPAWN_BORDER)
        goal_fine = False
        distance = 0
        while not goal_fine:
            goal_fine = True
            distance = math.sqrt((self.goal_x-self.x)**2+(self.goal_y-self.y)**2)
            if distance<MIN_START_DIST:
                goal_fine=False
                self.goal_x = np.random.uniform(0, self.basis.size.width)
                self.goal_y = np.random.uniform(0, self.basis.size.height)
            for i in range(len(self.basis.obstacles)):
                if self.basis.obstacles[i].Loc.x-self.basis.obstacles[i].Size.width/2<self.goal_x<self.basis.obstacles[i].Loc.x+self.basis.obstacles[i].Size.width/2 and self.basis.obstacles[i].Loc.y-self.basis.obstacles[i].Size.height/2<self.goal_y<self.basis.obstacles[i].Loc.y+self.basis.obstacles[i].Size.height/2:
                    goal_fine = False
                    self.goal_x = np.random.uniform(0, self.basis.size.width)
                    self.goal_y = np.random.uniform(0, self.basis.size.height)
                    break
        self.goal_x = self.goal_x-self.basis.size.width/2
        self.goal_y = self.goal_y-self.basis.size.height/2
        self.goal = p.loadURDF("goal.urdf", [self.goal_x, self.goal_y, 0.1])

    def reset_situation(self, ideal_angle, new_goal=True):
        x = 0
        y = 0
        position_fine = False
        while not position_fine:
            x = np.random.uniform(SPAWN_BORDER, self.basis.size.width-SPAWN_BORDER)
            y = np.random.uniform(SPAWN_BORDER, self.basis.size.height-SPAWN_BORDER)
            is_fine = True
            for i in range(len(self.basis.obstacles)):
                if self.basis.vobstacles[i].Loc.x < x < self.basis.vobstacles[i].Loc.x + self.basis.vobstacles[i].Size.width and self.basis.vobstacles[i].Loc.y < y < self.basis.vobstacles[i].Loc.y + self.basis.vobstacles[i].Size.height:
                    is_fine = False
                    break
            position_fine = is_fine
            if position_fine:
                break
        x=x-self.basis.size.width/2
        y=y-self.basis.size.height/2
        self.x = x
        self.y = y
        if (new_goal):
            self.new_goal()
        angle_to_goal = math.atan2(self.goal_y-self.y, self.goal_x-self.x)
        newangle = np.random.uniform(angle_to_goal-ideal_angle, angle_to_goal+ideal_angle)
        return newangle

    def step(self, action):
        target = False
        done = False
        achieved_goal = False

        p.stepSimulation()

        if GUI:
            time.sleep(TIME_DELTA)

        position, quaternion = p.getBasePositionAndOrientation(self.robotid)
        if position[2] < -1:
            p.resetBasePositionAndOrientation(self.robotid, [position[0], position[1], 0], quaternion)
        if position[0] < -self.basis.size.width/2 or position[0] > self.basis.size.width/2 or position[1] < -self.basis.size.height/2 or position[1] > self.basis.size.height/2:
            p.resetBasePositionAndOrientation(self.robotid, [0,0, position[2]], quaternion)
            done = True
        if math.isnan(position[0]) or math.isnan(position[1]) or math.isnan(position[2]):
            p.resetBasePositionAndOrientation(self.robotid, [0,0, position[2]], quaternion)
            done = True
        p.resetBasePositionAndOrientation(self.robotid, [position[0], position[1], 0.25], quaternion)

        dist_traveled = math.sqrt((position[0]-self.x)**2+(position[1]-self.y)**2)

        self.x = position[0]
        self.y = position[1]
        distance = math.sqrt((self.goal_x-self.x)**2+(self.goal_y-self.y)**2)
        q = Quaternion()
        q.define(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
        quaternion = q
        roll, pitch, yaw = quaternion.to_euler()

        if type(action) == torch.Tensor:
            action = action.cpu().detach().numpy()

        action_x = 1*MAX_SPEED*math.cos(yaw)
        action_y = 1*MAX_SPEED*math.sin(yaw)

        rotation_yaw = action[0]*float(MAX_ANGULAR_SPEED)

        p.resetBaseVelocity(self.robotid, [action_x, action_y, 0], [0, 0, rotation_yaw])

        collision = self.get_collisions()

        if roll > math.radians(TIP_ANGLE) or roll < math.radians(-TIP_ANGLE) or pitch > math.radians(TIP_ANGLE) or pitch < math.radians(-TIP_ANGLE):
            collision = True

        try:
            angle = round(yaw)
        except Exception as e:
            angle = 0

        distance = np.linalg.norm(
            [self.x - self.goal_x, self.y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.x
        skew_y = self.goal_y - self.y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        try:
            beta = math.acos(dot / (mag1 * mag2))
        except:
            print("Divide by zero error in beta calculation")
            beta = 0
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta


        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            achieved_goal = True
            done = True
        robot_state = [self.x, self.y, self.goal_x, self.goal_y, angle, self.gangle, self.run_lidar()]
        #reward = self.get_reward(target, collision, action)
        #return robot_state, reward, done, target
        return robot_state, collision, done, achieved_goal, dist_traveled, theta


    def reset(self, ideal_angle=np.pi): # Create a new environment
        p.resetSimulation()
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)
        self.environment_ids = []
        self.environment_dim = 0

        self.initial_state = ObjectState()
        self.initial_state.position = [0, 0, 0.1]
        self.initial_state.orientation = [0, 0, 0, 1]

        self.floor = p.loadURDF("floor.urdf")
        self.environment_ids.append(self.floor)
        for i in range(4):
            self.environment_ids.append(p.loadURDF("bound"+str(i+1)+".urdf"))
        for i in range(len(self.basis.obstacles)):
            self.obstacles.append(p.loadURDF("obs_"+str(i+1)+".urdf"))
            self.environment_ids.append(self.obstacles[i])
        for i in range(len(self.environment_ids)):
            self.environment_dim += p.getNumJoints(self.environment_ids[i])
        print(self.environment_dim)

        angle = self.reset_situation(ideal_angle)


        quaternion = Quaternion.from_euler(0, 0, angle)
        obj_state = self.initial_state


        obj_state.position = [self.x, self.y, 0.25]
        obj_state.orientation = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        self.robotid = p.loadURDF("robot.urdf", obj_state.position, obj_state.orientation)


        if GUI:
            time.sleep(TIME_DELTA)

        skew_x = self.goal_x - self.x
        skew_y = self.goal_y - self.y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        distance = math.sqrt((self.goal_x-self.x)**2+(self.goal_y-self.y)**2)

        robot_state = [self.x, self.y, self.goal_x, self.goal_y, angle, self.gangle, self.run_lidar()]
        return robot_state, distance, theta

    def reload(self, state, ideal_angle=np.pi):
        p.resetSimulation()
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(0.01)
        p.setTimeStep(TIME_DELTA)

        for i in range(4):
            self.environment_ids.append(p.loadURDF("bound"+str(i+1)+".urdf"))
        for i in range(len(self.basis.obstacles)):
            self.obstacles.append(p.loadURDF("obs_"+str(i+1)+".urdf"))
            self.environment_ids.append(self.obstacles[i])
        for i in range(len(self.environment_ids)):
            self.environment_dim += p.getNumJoints(self.environment_ids[i])

        angle = self.reset_situation(ideal_angle, new_goal=False)
        quaternion = Quaternion.from_euler(0, 0, angle)
        self.x = state[1]
        self.y = state[2]
        self.initial_state = ObjectState()
        obj_state = self.initial_state
        obj_state.position = [self.x, self.y, 0.25]
        obj_state.orientation = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        self.robotid = p.loadURDF("robot.urdf", obj_state.position, obj_state.orientation)

        self.goal_x = state[3]
        self.goal_y = state[4]

        self.goal = p.loadURDF("goal.urdf", [self.goal_x, self.goal_y, 0.25])

        skew_x = self.goal_x - self.x
        skew_y = self.goal_y - self.y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        distance = math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)

        robot_state = [theta, self.x, self.y, self.goal_x, self.goal_y, 0.0, self.run_lidar()]
        return robot_state, distance, theta




    @staticmethod
    def get_reward(target, collision, action):
        if target:
            return 100.0
        elif collision:
            return -100
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2

    def get_collisions(self):
        points = p.getContactPoints(self.robotid)
        filtered_points = []
        for point in points:
            if point[1] != self.floor and point[2] != self.floor:
                filtered_points.append(point)
        if len(filtered_points) > 0:
            return True
        else:
            return False

    def run_lidar(self):
        for did in self.ray_debug_id:
            p.removeUserDebugItem(did)
        self.ray_debug_id.clear()
        start_positions = []
        end_positions = []
        for i in range(LIDAR_POINTS):
            start_positions.append([self.x, self.y, 0.5])
            end_positions.append([self.x + LIDAR_RANGE * math.cos(i * 2 * math.pi / LIDAR_POINTS - LIDAR_ANGLE/2),
                                  self.y + LIDAR_RANGE * math.sin(i * 2 * math.pi / LIDAR_POINTS - LIDAR_ANGLE/2), 0.5])
        res = p.rayTestBatch(start_positions, end_positions)

        distances = []
        non_goal = []

        for i in range(len(res)):
            result = res[i]
            if result[0] > -1:
                hit_position = result[3]  # Get the collision point
                start_point = start_positions[i]
                distance_to_collision = math.sqrt(math.pow(hit_position[0] - start_point[0], 2) + math.pow(hit_position[1] - start_point[1], 2))
                distances.append(distance_to_collision)
                if result[0] != self.goal:
                    non_goal.append(distance_to_collision)
                #debug_ray = p.addUserDebugLine(start_point, hit_position, [1, 0, 0], 1, 0.01)
                #self.ray_debug_id.append(debug_ray)
            #else:
                #debug_ray = p.addUserDebugLine(start_positions[i], end_positions[i], [0, 1, 0], 1, 0.01)
                #self.ray_debug_id.append(debug_ray)

        self.lidar_dists = distances

        if len(non_goal) == 0:
            return 0
        return min(non_goal)