import math
import wandb
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from colour import Color
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
            episodes,
            plotter_display=True
    ):
        if plotter_display is True:
            plt.ion()
        self.prev_total_reward = 0
        self.avg_x = np.arange(episodes)
        self.avg_y = np.zeros(episodes)
        self.dist_x = np.arange(episodes)
        self.dist_y = np.zeros(episodes)
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
        self.c_loss_x = np.arange(episodes)
        self.c_loss_y = np.zeros(episodes)
        self.a_loss_x = np.arange(episodes)
        self.a_loss_y = np.zeros(episodes)

        if plotter_display is True:
            self.fig, self.ax = plt.subplots(7, figsize=(10,6))
            self.fig.suptitle("Training Metrics")
            self.avg_line, = self.ax[0].plot(self.avg_x, self.avg_y, label="Average Reward", color="blue")
            self.dist_line, = self.ax[1].plot(self.dist_x, self.dist_y, label="Distance Reward", color="red")
            self.total_reward_line, = self.ax[2].plot(self.total_reward_x, self.total_reward_y, label="Total Reward", color="green")
            self.dist_weight_line, = self.ax[3].plot(self.dist_weight_x, self.dist_weight_y, label="Anglevel Weight", color="red")
            self.angle_weight_line, = self.ax[3].plot(self.angle_weight_x, self.angle_weight_y, label="Mindist Weight", color="blue")
            self.time_weight_line, = self.ax[3].plot(self.time_weight_x, self.time_weight_y, label="Velocity Weight", color="green")
            self.achieve_line, = self.ax[4].plot(self.achieve_chance_x, self.achieve_chance_y, label="Achieve Chance", color="green")
            self.collision_line, = self.ax[4].plot(self.collision_chance_x, self.collision_chance_y, label="Collision Chance", color="red")
            self.none_line, = self.ax[4].plot(self.none_chance_x, self.none_chance_y, label="None Chance", color="blue")
            self.a_loss_line = self.ax[5].plot(self.a_loss_x, self.a_loss_y, label="Actor Loss", color="blue")[0]
            self.c_loss_line = self.ax[6].plot(self.c_loss_x, self.c_loss_y, label="Critic Loss", color="red")[0]
            self.ax[3].legend(handles=[self.dist_weight_line, self.angle_weight_line, self.time_weight_line])
            self.ax[4].legend(handles=[self.achieve_line, self.collision_line, self.none_line])
        self.show = plotter_display
        if plotter_display is True:
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
        c_loss_y,
        a_loss_y,
        eval_rew,
        eval_ac
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

        if type(dist_weight_y) == torch.Tensor and type(angle_weight_y) == torch.Tensor:
            self.dist_weight_y.put(episode, dist_weight_y.cpu().detach().numpy())
            self.angle_weight_y.put(episode, angle_weight_y.cpu().detach().numpy())
        else:
            self.dist_weight_y.put(episode, dist_weight_y)
            self.angle_weight_y.put(episode, angle_weight_y)
        self.time_weight_y.put(episode, time_weight_y)

        self.achieve_history.put(episode, achieve_chance_y)
        self.collision_history.put(episode, collision_chance_y)
        if achieve_chance_y == 0 and collision_chance_y == 0:
            self.none_history.put(episode, 1)

        achieve_prob = 100*sum(self.achieve_history) / (episode + 1)
        collision_prob = 100*sum(self.collision_history) / (episode + 1)
        if episode > 100:
            recent_achieves = np.zeros(100)
            recent_collisions = np.zeros(100)
            for i in range(100):
                recent_achieves.put(i, self.achieve_history[episode-i])
                recent_collisions.put(i, self.collision_history[episode-i])
            achieve_prob = sum(recent_achieves)
            collision_prob = sum(recent_collisions)

            total = np.zeros(episode+1)
            for i in range(episode+1):
                if len(recent_achieves) == 0:
                    break
                elif recent_achieves[i] == 1:
                    total[i] = 0
                    break
                elif recent_collisions[i] == 1:
                    total[i] = 0
                    break
                else:
                    total[i] = 1


            none_prob = 100-sum(total)
        total = np.zeros(episode + 1)
        for i in range(episode+1):
            if self.achieve_history[i] == 1:
                total[i] = 0
                break
            elif self.collision_history[i] == 1:
                total[i] = 0
                break
            else:
                total[i] = 1

        self.achieve_chance_y.put(episode, achieve_prob)
        self.collision_chance_y.put(episode, collision_prob)
        self.none_chance_y.put(episode, 100-achieve_prob-collision_prob)

        self.c_loss_y.put(episode, c_loss_y)
        self.a_loss_y.put(episode, a_loss_y)

        if self.show is True:
            self.total_reward_line.set_data(self.total_reward_x, self.total_reward_y)
            self.avg_line.set_data(self.avg_x, self.avg_y)
            self.dist_line.set_data(self.dist_x, self.dist_y)
            self.dist_weight_line.set_data(self.dist_weight_x, self.dist_weight_y)
            self.angle_weight_line.set_data(self.angle_weight_x, self.angle_weight_y)
            self.time_weight_line.set_data(self.time_weight_x, self.time_weight_y)
            self.achieve_line.set_data(self.achieve_chance_x, self.achieve_chance_y)
            self.collision_line.set_data(self.collision_chance_x, self.collision_chance_y)
            self.none_line.set_data(self.none_chance_x, self.none_chance_y)
            self.c_loss_line.set_data(self.c_loss_x, self.c_loss_y)
            self.a_loss_line.set_data(self.a_loss_x, self.a_loss_y)

        try:
            if self.show is True:
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
                self.ax[5].set_xlim(0, episode+1)
                closs_max = np.max(self.c_loss_y)
                aloss_max = np.max(self.a_loss_y)
                closs_min = np.min(self.c_loss_y)
                aloss_min = np.min(self.a_loss_y)
                self.ax[6].set_ylim(closs_min, closs_max)
                self.ax[6].set_xlim(0, episode+1)
                self.ax[5].set_ylim(aloss_min, aloss_max)
        except:
            print("Error in setting limits")


        if eval_rew == -1 and eval_ac == -1:
            wandb_params = {
                "actor_loss" : self.a_loss_y[episode],
                "critic_loss" : self.c_loss_y[episode],
                "avg_reward" : self.avg_y[episode],
                "achieve_rate" : self.achieve_chance_y[episode],
                "none_rate" : self.none_chance_y[episode],
                "collision_rate" : self.collision_chance_y[episode],
                "anglevel_reward" : self.dist_weight_y[episode],
                "vel_reward" : self.time_weight_y[episode]
            }
        else:
            wandb_params = {
                "actor_loss" : self.a_loss_y[episode],
                "critic_loss" : self.c_loss_y[episode],
                "avg_reward" : self.avg_y[episode],
                "achieve_rate" : self.achieve_chance_y[episode],
                "none_rate": self.none_chance_y[episode],
                "collision_rate": self.collision_chance_y[episode],
                "anglevel_reward" : self.dist_weight_y[episode],
                "vel_reward" : self.time_weight_y[episode],
                "eval_reward" : eval_rew,
                "eval_achieve_chance" : eval_ac
            }
        wandb.log(wandb_params)

        if self.show is True:
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
        self.a_loss_y = np.asarray(stats["actor_loss_y"])
        self.c_loss_y = np.asarray(stats["critic_loss_y"])


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
            "none_chance_y": self.none_chance_y,
            "actor_loss" : self.a_loss_y,
            "critic_loss" : self.c_loss_y,
            "avg_reward" : self.avg_y
        }
        return stats

    def get_ideal_probability(self, ideal_angle_episode, episode):
        if ideal_angle_episode>99:
            recent_achieves = self.achieve_history
            for i in range(episode):
                if i < episode-100:
                    recent_achieves = np.delete(recent_achieves, 0)
                    recent_achieves = np.resize(recent_achieves, recent_achieves.size -1)
                else:
                    break
            return sum(recent_achieves)/100
        else:
            return 0

    def get_achieve_chance(self, episode):
        return self.achieve_chance_y[episode]

class Model_Visualizer:
    def __init__(self, sizex, sizey):
        plt.ion()
        self.fig = plt.figure(figsize=(sizex, sizey))
        self.ax = self.fig.subplots()
        self.sizex = sizex
        self.sizey = sizey
        self.x = 0
        self.y = 0
        self.goalx = 0
        self.goaly = 0
        self.ax.set_xlim(-self.sizex/2, self.sizex/2)
        self.ax.set_ylim(-self.sizey/2, self.sizey/2)
    def start(self, x, y, goalx, goaly):
        self.goalx = goalx
        self.goaly = goaly
        self.x = x
        self.y = y
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    def update(self, x, y, q):
        q1, q2 = q
        qvals = []
        for i in range(len(q1)):
            qvals.append(min(q1[i], q2[i]).item())
        grad = list(Color("red").range_to(Color("green"), len(qvals)))
        ordered_q = sorted(qvals)
        for i in range(len(qvals)):
            self.ax.scatter(x[i], y[i], color=grad[ordered_q.index(qvals[i])].hex)
        self.ax.scatter(self.goalx, self.goaly, color='purple')
        self.ax.scatter(self.x, self.y, color='blue')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    def clear(self):
        self.ax.clear()
        self.ax.set_xlim(-self.sizex/2, self.sizex/2)
        self.ax.set_ylim(-self.sizey/2, self.sizey/2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action.cpu().detach().numpy() + ou_state


class Normalizer:
    def __init__(self, map):
        self.map = map
        self.max_mes = math.sqrt(self.map.size.width**2+self.map.size.height**2)

    def NormalizeState(self, state):
        '''
        mem_t = Transition(*zip(*mem.memory))
        nstate = []
        states = torch.stack(mem_t.state).cpu()
        column_means = states.mean(dim=0)
        column_stddevs = states.std(dim=0) + 1e-8 * torch.abs(column_means)
        if torch.isnan(column_stddevs).any():
            return state
        nstate.append((state[0] - column_means[0]) / column_stddevs[0])
        location_states_means = [column_means[1], column_means[2], column_means[3], column_means[4]]
        location_states_std = [column_stddevs[1], column_stddevs[2], column_stddevs[3], column_stddevs[4]]
        nstate.append((state[1] - location_states_means[0]) / (location_states_std[0]))
        nstate.append((state[2] - location_states_means[1]) / (location_states_std[1]))
        nstate.append((state[3] - location_states_means[2]) / (location_states_std[2]))
        nstate.append((state[4] - location_states_means[3]) / (location_states_std[3]))
        if column_stddevs[5] == 0 or column_stddevs[6] == 0:
            nstate.append(state[5])
            nstate.append(state[6])
            nstate.append(state[7])
            nstate.append(state[8])
            return nstate
        nstate.append((state[5] - column_means[5]) / column_stddevs[5])
        nstate.append((state[6] - column_means[6]) / column_stddevs[6])
        nstate.append(state[7])
        nstate.append(state[8])
        return nstate'''
        nstate = [
            (state[0]/np.pi),
            (2*state[1]/self.map.size.width),
            (2*state[2]/self.map.size.height),
            (2*state[3]/self.map.size.width),
            (2*state[4]/self.map.size.height),
            state[5]*2/self.max_mes,
            state[6]*2/self.max_mes,
            state[7],
            state[8]
        ]
        return torch.FloatTensor(nstate)

    def NormalizeStates(self, mem, states):
        nstates = []
        for i in range(len(states)):
            nstates.append(self.NormalizeState(states[i]))
        return nstates
