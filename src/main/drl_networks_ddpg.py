import torch
import numpy as np
import torch.nn as nn
import time
import os
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from drl_utils import ReplayMemory, Transition, NumpyArrayEncoder, Model_Plotter, OUNoise, Model_Visualizer
from drl_venv import MAX_SPEED, TIME_DELTA
from IPython import display
import torch.optim as optim
import random

FILE_LOCATION = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\models"
FILE_NAME = "SampleModel"
SAVE_FREQ = 100

EPISODES = 16000
MAX_TIMESTEP = 1000
BATCH_SIZE = 128

COLLISION_WEIGHT = -100
TIME_WEIGHT = 0#-6
FINISH_WEIGHT = 100
DIST_WEIGHT = 2
PASS_DIST_WEIGHT = -6
CHALLENGE_WEIGHT = 10
CHALLENGE_EXP_BASE = 1.25
ANGLE_WEIGHT = -6#-2
SPEED_WEIGHT = 3

STATE_DIM = 9
ACTION_DIM = 2

ACTOR_LAYER_1 = 512
ACTOR_LAYER_2 = 256

ACTOR_LR = 1e-4

CRITIC_LAYER_1 = 512
CRITIC_LAYER_2 = 512

CRITIC_LR = 1e-5

START_WEIGHT_THRESHOLD = 3e-3
GAMMA = 0.99
TAU = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, ACTOR_LAYER_1)
        self.l2 = nn.Linear(ACTOR_LAYER_1, ACTOR_LAYER_2)
        self.l3 = nn.Linear(ACTOR_LAYER_2, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = state.to(torch.float32)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = self.tanh(a)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, CRITIC_LAYER_1)
        self.l2 = nn.Linear(CRITIC_LAYER_1, CRITIC_LAYER_2)
        self.l3 = nn.Linear(CRITIC_LAYER_2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        x = x.to(torch.float32)
        a = a.to(torch.float32)
        q = self.l1(torch.cat([x,a], 1))
        q = self.relu(q)
        q = self.l2(q)
        q = self.relu(q)
        return self.l3(q)

class DDPG(object):
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.mem = ReplayMemory(1000000)
        self.criterion = nn.MSELoss()

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state):
        action = self.actor.forward(state)
        return action

    def update_parameters(self, batch_size):
        if len(self.mem) < batch_size:
            return
        batch = Transition(*zip(*self.mem.sample(batch_size)))
        states = torch.stack(batch.state).to(self.device).float()
        actions = torch.stack(batch.action).to(self.device).float()
        rewards = torch.stack(batch.reward).to(self.device).float()
        next_states = torch.stack(batch.next_state).to(self.device).float()

        #Critic loss
        QVal = self.critic.forward((states, actions))
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward((next_states, next_actions.detach()))
        QPrime = rewards + GAMMA * next_Q
        critic_loss = self.criterion(QVal, QPrime)
        critic_loss = critic_loss.float()

        #Actor loss
        actor_loss = -self.critic.forward((states, self.actor.forward(states))).mean()

        #Update networks
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_update(self.actor_target, self.actor, TAU)
        self.soft_update(self.critic_target, self.critic, TAU)

class TrainingExecutor:
    def __init__(self):
        self.ddpg = DDPG(STATE_DIM, ACTION_DIM, DEVICE)
        self.plotter = None

    def get_reward(self, done, collision, timestep, achieved_goal, delta_dist, initdistance, initangle, ovr_dist, angle, vel):
        dist_weight = delta_dist * DIST_WEIGHT
        angle_weight = (abs(angle) / np.pi) * ANGLE_WEIGHT
        # time_weight = TIME_WEIGHT*(initdistance/MAX_SPEED-timestep/TIME_DELTA)
        time_weight = 0
        speed_weight = abs(vel) * SPEED_WEIGHT
        total_weight = dist_weight + angle_weight + time_weight + speed_weight
        if (ovr_dist > initdistance):
            total_weight += PASS_DIST_WEIGHT
        if initdistance / MAX_SPEED < timestep * TIME_DELTA:
            total_weight += TIME_WEIGHT
            time_weight = TIME_WEIGHT
        if done:
            if achieved_goal:
                initangle = abs(initangle) / np.pi
                dist_challenge = CHALLENGE_EXP_BASE ** initdistance
                angle_challenge = CHALLENGE_EXP_BASE ** initangle
                return FINISH_WEIGHT + time_weight + dist_challenge * CHALLENGE_WEIGHT + angle_challenge * CHALLENGE_WEIGHT, time_weight, dist_weight, angle_weight
            elif collision is not True:
                return COLLISION_WEIGHT + time_weight, time_weight, dist_weight, angle_weight
        if collision is True:
            return COLLISION_WEIGHT + time_weight, time_weight, dist_weight, angle_weight
        return total_weight, time_weight, dist_weight, angle_weight

    def train(self, env, num_episodes=EPISODES, max_steps=MAX_TIMESTEP, batch_size=BATCH_SIZE, start_episode=0):
        noise = OUNoise(ACTION_DIM)
        rewards = []
        if self.plotter is None:
            self.plotter = Model_Plotter(num_episodes)
        visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)
        for episode in range(start_episode, num_episodes):
            state, dist = env.reset()
            visualizer.start(state[1], state[2], state[3], state[4])
            initdist = dist
            initangle = state[1]
            state = torch.FloatTensor(state).to(DEVICE)
            noise.reset()
            episode_reward = 0
            episode_tw = 0
            episode_dw = 0
            episode_aw = 0
            episode_achieve = 0
            episode_collide = 0
            episode_x = []
            episode_y = []
            action_q = []
            ovr_dist = 0

            states = []
            actions = []

            for step in range(max_steps):
                action = self.ddpg.get_action(state)
                action = noise.get_action(action, step)
                actions.append(torch.tensor(action))
                states.append(state)
                next_state, collision, done, achieved_goal, dist_traveled = env.step(action, step)
                ovr_dist += dist_traveled
                reward, tw, dw, aw = self.get_reward(done, collision, step, achieved_goal, dist_traveled, initdist, initangle, ovr_dist, next_state[0], action[1])
                self.ddpg.mem.push(state, torch.tensor(action).to(DEVICE), torch.tensor(next_state).to(DEVICE), torch.tensor([reward]).to(DEVICE))
                episode_reward += reward
                episode_tw += tw
                episode_dw += dw
                episode_aw += aw
                episode_x.append(next_state[1])
                episode_y.append(next_state[2])
                state = torch.FloatTensor(next_state).to(DEVICE)

                self.ddpg.update_parameters(batch_size)

                if collision is True:
                    episode_collide = 1
                    done = True
                if achieved_goal is True:
                    episode_achieve = 1
                if done:
                    break

            states = torch.stack(states).to(self.ddpg.device)
            actions = torch.stack(actions).to(self.ddpg.device)
            action_q = self.ddpg.critic.forward((states, actions)).detach().cpu().numpy()
            visualizer.clear()
            visualizer.update(episode_x, episode_y, action_q)
            if SAVE_FREQ != -1 and episode % SAVE_FREQ == 0:
                self.save(episode)

            rewards.append(episode_reward)
            print("Episode: " + str(episode) + " Reward: " + str(episode_reward))
            self.plotter.update(episode, initdist, episode_reward, episode_dw, episode_aw, episode_tw, episode_achieve, episode_collide)



    def save(self, episode, filename=FILE_NAME, directory=FILE_LOCATION):
        if not os.path.exists(directory + "\\" + filename):
            os.mkdir(directory + "\\" + filename)
        if self.plotter is not None:
            statistics = self.plotter.save()
        else:
            statistics = {}
        with open(f"{directory}/{filename}/statistics.json", "w") as outfile:
            json.dump(statistics, outfile, cls=NumpyArrayEncoder)
        with open(f"{directory}/{filename}/memory.json", "w") as outfile:
            outfile.write(self.ddpg.mem.to_json())
        with open(f"{directory}/{filename}/core.json", "w") as outfile:
            json.dump({
                "episode": episode
            }, outfile)
        torch.save(self.ddpg.actor.state_dict(), f"{directory}/{filename}/agent_actor.pth")
        torch.save(self.ddpg.critic.state_dict(), f"{directory}/{filename}/agent_critic.pth")

    def load(self, episodes=EPISODES, filename=FILE_NAME, directory=FILE_LOCATION):
        with open(f"{directory}/{filename}/statistics.json", "r") as infile:
            stats = json.load(infile)
            self.plotter = Model_Plotter(episodes)
            np_stats = {
                "avg_y": np.asarray(stats["avg_y"]),
                "dist_y": np.asarray(stats["dist_y"]),
                "total_reward_y": np.asarray(stats["total_reward_y"]),
                "dist_weight_y": np.asarray(stats["dist_weight_y"]),
                "angle_weight_y": np.asarray(stats["angle_weight_y"]),
                "time_weight_y": np.asarray(stats["time_weight_y"]),
                "achieve_history": np.asarray(stats["achieve_history"]),
                "collision_history": np.asarray(stats["collision_history"]),
                "none_history": np.asarray(stats["none_history"]),
                "achieve_chance_y": np.asarray(stats["achieve_chance_y"]),
                "collision_chance_y": np.asarray(stats["collision_chance_y"]),
                "none_chance_y": np.asarray(stats["none_chance_y"])
            }
            self.plotter.load(np_stats)
        with open(f"{directory}/{filename}/memory.json", "r") as infile:
            self.ddpg.mem.from_json(infile.read(), DEVICE)
        with open(f"{directory}/{filename}/core.json", "r") as infile:
            episode = json.load(infile)["episode"]
        self.ddpg.actor.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.ddpg.critic.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic.pth"))
        return episode