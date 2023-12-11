import lightning
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import sys
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from rewards import ProgressiveRewards, SimpleReward
from collections import namedtuple
from drl_venv import RobotVEnv
from generate_urdf_file_map import from_map, ASSET_PATH
from map import Map
from torch.utils.data import DataLoader


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, state):
        state = state.to(dtype=torch.float32)
        a = self.l1(state)
        a = self.relu(a)
        a = self.l2(a)
        a = self.relu(a)
        a = self.l3(a)
        a = self.tanh(a)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2_s = nn.Linear(400, 300)
        self.l2_a = nn.Linear(action_dim, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim, 400)
        self.l5_s = nn.Linear(400, 300)
        self.l5_a = nn.Linear(action_dim, 300)
        self.l6 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        state = state.to(dtype=torch.float32)
        action = action.to(dtype=torch.float32)
        q1 = self.q1(state, action)
        s2 = self.relu(self.l4(state))
        self.l5_s(s2)
        self.l5_a(action)
        s21 = torch.mm(s2, self.l5_s.weight.t())
        s22 = torch.mm(action, self.l5_a.weight.t())
        s2 = self.relu(s21 + s22 + self.l5_a.bias.data)
        q2 = self.l6(s2)

        return q1.to(dtype=torch.float16), q2.to(dtype=torch.float16)

    def q1(self, state, action):
        s1 = self.relu(self.l1(state))
        self.l2_s(s1)
        self.l2_a(action)
        s11 = torch.mm(s1, self.l2_s.weight.t())
        s12 = torch.mm(action, self.l2_a.weight.t())
        s1 = self.relu(s11 + s12 + self.l2_a.bias.data)
        q1 = self.l3(s1)
        return q1


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


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

    def get(self):
        return self.memory


class PLite():
    def __init__(self, device="cpu", epochs=100, mixed_precision=False, cudnn=False):
        if cudnn:
            torch.backends.cudnn.benchmark = True
        self.device = device
        self.epochs = epochs
        self.mixed_precision = mixed_precision

        print("Created PLite model\n"
              "Model Info: \n"
              " - Device: " + str(self.device) + "\n"
                " - Epochs: " + str(self.epochs) + "\n"
                " - Mixed Precision: " + str(self.mixed_precision) + "\n"
                " - CUDNN: " + str(cudnn) + "\n")

    def train_step(self, **args):
        """
        Trains one epoch of the model
        :param args:
        :return: none
        """
        pass


class PLiteRunner():
    def __init__(self, plite_model, environment):
        if type(plite_model) != PLite:
            raise TypeError("plite_model must be of type PLite")
        if type(environment) != RobotVEnv:
            raise TypeError("environment must be of type RobotVEnv")
        print("PLiteRunner initialized: ")
        self.environment = environment
        self.plite_model = plite_model

    def run(self, **args):
        """
        Runs the model
        :param args:
        :return: none
        """
        print("Started running model")
        for epoch in range(self.plite_model.epochs):
            print("Epoch: " + str(epoch + 1))
            self.plite_model.train_step(environment=self.environment, **args)


class TD3(PLite):
    def __init__(self, device="cpu", epochs=100, mixed_precision=False, cudnn=False):
        super().__init__(device=device, epochs=epochs, mixed_precision=mixed_precision, cudnn=cudnn)
        self.state_dim = 10
        self.action_dim = 2

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def train_step(self, environment=None, **args):
        if environment is None:
            raise ValueError("environment cannot be None")



if __name__ == "__main__":
    if sys.platform == "win32":
        path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\BasicMap.json'
        logger_path = "G:\Projects\AutoNav\AutoNavServer\output\logs"
    elif sys.platform == "linux" or sys.platform == "linux2":
        path = '/home/jovyan/workspace/AutoNavServer/assets/testing/FRC2023Map.json'
        wandb_path = '/home/jovyan/workspace/AutoNavServer/wandb'
        logger_path = '/home/jovyan/workspace/AutoNavServer/output/logs'
    elif sys.platform == "darwin":
        path = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/BasicMap.json"
        logger_path = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/output/logs"
    else:
        print("SYSTEM NOT SUPPORTED. EXITING")
        exit()
    env = RobotVEnv(map=Map(json.load(open(path))), assets_path=ASSET_PATH)
    model = TD3()
    runner = PLiteRunner(model, environment=env)
    runner.run()
