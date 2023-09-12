import numpy as np
import random
import json
import torch
import matplotlib.pyplot as plt
from json import JSONEncoder
from collections import namedtuple

class ObjectState:
    def __init__(self):
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 0]
        self.linear_velocity = [0, 0, 0]
        self.angular_velocity = [0, 0, 0]

class Quaternion:
    def __init__(self):
        self.w = 0
        self.x = 0
        self.y = 0
        self.z = 1
    def define(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_euler(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = Quaternion()

        q.define(cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)

        return q

    def to_euler(self):
        roll = np.arctan2(2*(self.w*self.x+self.y*self.z), 1-2*(self.x**2+self.y**2))
        pitch = np.arcsin(2*(self.w*self.y-self.z*self.x))
        yaw = np.arctan2(2*(self.w*self.z+self.x*self.y), 1-2*(self.y**2+self.z**2))

        return roll, pitch, yaw

import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Save a transition"""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def to_json(self):
        json_dict = {
            "capacity": self.capacity,
            "memory": self.memory
        }
        return json.dumps(json_dict, indent=4, cls=TensorEncoder)

    def from_json(self, json_obj, device):
        json_dict = json.loads(json_obj)
        self.capacity = json_dict["capacity"]
        for i in range(len(json_dict["memory"])):
            state = torch.tensor(json_dict["memory"][i][0]).to(device)
            action = torch.tensor(json_dict["memory"][i][1]).to(device)
            if (json_dict["memory"][i][2] == None):
                next_state = None
            else: next_state = torch.tensor(json_dict["memory"][i][2]).to(device)
            reward = torch.tensor(json_dict["memory"][i][3]).to(device)
            self.memory.append(Transition(state, action, next_state, reward))

    def __len__(self):
        return len(self.memory)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class TensorEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

class Model_Plotter():
    def __init__(
            self,
            episodes
    ):
        plt.ion()
        self.prev_total_reward = 0
        self.avg_x = np.arange(episodes)
        self.avg_y = np.zeros(episodes)
        self.dist_x = np.arange(episodes)
        self.dist_y = np.empty(episodes)
        self.total_reward_x = np.arange(100)
        self.total_reward_y = np.zeros(100)
        self.dist_weight_x = np.arange(episodes)
        self.dist_weight_y = np.zeros(episodes)
        self.angle_weight_x = np.arange(episodes)
        self.angle_weight_y = np.zeros(episodes)
        self.time_weight_x = np.arange(episodes)
        self.time_weight_y = np.zeros(episodes)
        self.achieve_history = np.zeros(episodes)
        self.collision_history = np.zeros(episodes)
        self.none_history = np.zeros(episodes)
        self.achieve_chance_x = np.arange(episodes)
        self.achieve_chance_y = np.zeros(episodes)
        self.collision_chance_x = np.arange(episodes)
        self.collision_chance_y = np.zeros(episodes)
        self.none_chance_x = np.arange(episodes)
        self.none_chance_y = np.zeros(episodes)

        self.fig, self.ax = plt.subplots(5, figsize=(10,6))
        self.fig.suptitle("Training Metrics")
        self.avg_line, = self.ax[0].plot(self.avg_x, self.avg_y, label="Average Reward", color="blue")
        self.dist_line, = self.ax[1].plot(self.dist_x, self.dist_y, label="Distance Reward", color="red")
        self.total_reward_line, = self.ax[2].plot(self.total_reward_x, self.total_reward_y, label="Total Reward", color="green")
        self.dist_weight_line, = self.ax[3].plot(self.dist_weight_x, self.dist_weight_y, label="Distance Weight", color="red")
        self.angle_weight_line, = self.ax[3].plot(self.angle_weight_x, self.angle_weight_y, label="Angle Weight", color="blue")
        self.time_weight_line, = self.ax[3].plot(self.time_weight_x, self.time_weight_y, label="Time Weight", color="green")
        self.achieve_line, = self.ax[4].plot(self.achieve_chance_x, self.achieve_chance_y, label="Achieve Chance", color="green")
        self.collision_line, = self.ax[4].plot(self.collision_chance_x, self.collision_chance_y, label="Collision Chance", color="red")
        self.none_line, = self.ax[4].plot(self.none_chance_x, self.none_chance_y, label="None Chance", color="blue")
        self.ax[3].legend(handles=[self.dist_weight_line, self.angle_weight_line, self.time_weight_line])
        self.ax[4].legend(handles=[self.achieve_line, self.collision_line, self.none_line])

        plt.show()

    def update(
        self,
        episode,
        dist_y,
        total_reward_y,
        dist_weight_y,
        angle_weight_y,
        time_weight_y,
        achieve_chance_y,
        collision_chance_y,
    ):
        if self.total_reward_y[99]==self.prev_total_reward or episode > 99:
            self.total_reward_y = np.delete(self.total_reward_y, 0)
            self.total_reward_y = np.append(self.total_reward_y, 100)
            if type(total_reward_y) == torch.Tensor:
                self.total_reward_y.put(99, total_reward_y.cpu().detach().numpy())
                self.total_reward_x = np.arange(episode-99, episode+1)
                self.prev_total_reward = total_reward_y
            else:
                self.total_reward_y.put(99, total_reward_y)
                self.total_reward_x = np.arange(episode-99, episode+1)
                self.prev_total_reward = total_reward_y
        else:
            if type(total_reward_y) == torch.Tensor:
                self.total_reward_y.put(episode, total_reward_y.cpu().detach().numpy())
                self.total_reward_x = np.arange(episode-99, episode+1)
                self.prev_total_reward = total_reward_y
            else:
                self.total_reward_y.put(episode, total_reward_y)
                self.total_reward_x = np.arange(episode-99, episode+1)
                self.prev_total_reward = total_reward_y

        self.dist_y.put(episode, dist_y)

        avg_y = sum(self.total_reward_y) / (episode + 1)
        if episode>100:
            avg_y = sum(self.total_reward_y) / 100
        self.avg_y.put(episode, avg_y)

        self.dist_weight_y.put(episode, dist_weight_y.cpu().detach().numpy())
        self.angle_weight_y.put(episode, angle_weight_y.cpu().detach().numpy())
        self.time_weight_y.put(episode, time_weight_y)

        self.achieve_history.put(episode, achieve_chance_y)
        self.collision_history.put(episode, collision_chance_y)
        if achieve_chance_y == 0 and collision_chance_y == 0:
            self.none_history.put(episode, 1)

        achieve_prob = 100*sum(self.achieve_history) / (episode + 1)
        collision_prob = 100*sum(self.collision_history) / (episode + 1)
        if episode > 100:
            recent_achieves = self.achieve_history
            recent_collisions = self.collision_history
            for i in range(episode):
                if i < episode - 100:
                    recent_achieves = np.delete(recent_achieves, 0)
                    recent_achieves = np.resize(recent_achieves, recent_achieves.size - 1)
                    recent_collisions = np.delete(recent_collisions, 0)
                    recent_collisions = np.resize(recent_collisions, recent_achieves.size - 1)
                else:
                    break
            achieve_prob = sum(recent_achieves)
            collision_prob = sum(recent_collisions)

        none_prob = 100-achieve_prob-collision_prob
        self.achieve_chance_y.put(episode, achieve_prob)
        self.collision_chance_y.put(episode, collision_prob)
        self.none_chance_y.put(episode, none_prob)


        self.total_reward_line.set_data(self.total_reward_x, self.total_reward_y)
        self.avg_line.set_data(self.avg_x, self.avg_y)
        self.dist_line.set_data(self.dist_x, self.dist_y)
        self.dist_weight_line.set_data(self.dist_weight_x, self.dist_weight_y)
        self.angle_weight_line.set_data(self.angle_weight_x, self.angle_weight_y)
        self.time_weight_line.set_data(self.time_weight_x, self.time_weight_y)
        self.achieve_line.set_data(self.achieve_chance_x, self.achieve_chance_y)
        self.collision_line.set_data(self.collision_chance_x, self.collision_chance_y)
        self.none_line.set_data(self.none_chance_x, self.none_chance_y)

        self.ax[0].set_xlim(0, episode+1)
        self.ax[0].set_ylim(np.min(self.avg_y)-1, np.max(self.avg_y)+1)
        self.ax[1].set_xlim(0, episode+1)
        self.ax[1].set_ylim(np.min(self.dist_y), np.max(self.dist_y))
        self.ax[2].set_xlim(np.min(self.total_reward_x), np.max(self.total_reward_x))
        self.ax[2].set_ylim(np.min(self.total_reward_y)-1, np.max(self.total_reward_y)+1)
        self.ax[3].set_xlim(0, episode+1)
        self.ax[3].set_ylim(min(np.min(self.dist_weight_y), np.min(self.angle_weight_y), np.min(self.time_weight_y)), max(np.max(self.dist_weight_y), np.max(self.angle_weight_y), np.max(self.time_weight_y)))
        self.ax[4].set_xlim(0, episode+1)
        self.ax[4].set_ylim(0, 100)


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


    def load(self, stats):
        self.avg_y = np.asarray(stats["avg_y"])
        self.dist_y = np.asarray(stats["dist_y"])
        self.total_reward_y = np.asarray(stats["total_reward_y"])
        self.dist_weight_y = np.asarray(stats["dist_weight_y"])
        self.angle_weight_y = np.asarray(stats["angle_weight_y"])
        self.time_weight_y = np.asarray(stats["time_weight_y"])
        self.achieve_history = np.asarray(stats["achieve_history"])
        self.collision_history = np.asarray(stats["collision_history"])
        self.none_history = np.asarray(stats["none_history"])
        self.achieve_chance_y = np.asarray(stats["achieve_chance_y"])
        self.collision_chance_y = np.asarray(stats["collision_chance_y"])
        self.none_chance_y = np.asarray(stats["none_chance_y"])


    def save(self):
        stats = {
            "avg_y": self.avg_y,
            "dist_y": self.dist_y,
            "total_reward_y": self.total_reward_y,
            "dist_weight_y": self.dist_weight_y,
            "angle_weight_y": self.angle_weight_y,
            "time_weight_y": self.time_weight_y,
            "achieve_history": self.achieve_history,
            "collision_history": self.collision_history,
            "none_history": self.none_history,
            "achieve_chance_y": self.achieve_chance_y,
            "collision_chance_y": self.collision_chance_y,
            "none_chance_y": self.none_chance_y
        }
        return stats