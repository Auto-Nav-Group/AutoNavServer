import torch
import numpy as np
import torch.nn as nn
import time
import os
import sys
import json
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from drl_utils import ReplayMemory, Transition, NumpyArrayEncoder, Model_Plotter, OUNoise, Model_Visualizer, Normalizer
from drl_venv import MAX_SPEED, TIME_DELTA
from IPython import display
import torch.optim as optim
import random

SWEEP_CONFIG = {

}

if sys.platform == "win32":
    FILE_LOCATION = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\models"
elif sys.platform == "linux" or sys.platform == "linux2":
    FILE_LOCATION = "/home/jovyan/workspace/AutoNavServer/assets/drl/models"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
FILE_NAME = "SampleModel"
SAVE_FREQ = 4999
EVAL_FREQ = -1
VISUALIZER_ENABLED = True
SEPERATE_NORM_MEM = True

EPISODES = 5000
MAX_TIMESTEP = 100
BATCH_SIZE = 512

COLLISION_WEIGHT = -100
TIME_WEIGHT = 0#-6
FINISH_WEIGHT = 100
DIST_WEIGHT = 0
PASS_DIST_WEIGHT = 0
CHALLENGE_WEIGHT = 0.01
CHALLENGE_EXP_BASE = 1
ANGLE_WEIGHT = 0#-2
SPEED_WEIGHT = 0.5
ANGLE_SPEED_WEIGHT = -0.5#-0.5
MIN_DIST_WEIGHT = -0.5
WALL_DIST = 0

STATE_DIM = 9
ACTION_DIM = 2

ACTOR_LAYER_1 = 512
ACTOR_LAYER_2 = 512

ACTOR_LR = 0.000001
ACTOR_LR_STEP_SIZE = 180000
ACTOR_LR_GAMMA = 0.452
ACTOR_LR_WEIGHT_DECAY = 0.0001

CRITIC_LAYER_1 = 512
CRITIC_LAYER_2 = 512

CRITIC_LR = 0.00003
CRITIC_LR_STEP_SIZE = 96000
CRITIC_LR_GAMMA = 0.363
CRITIC_LR_WEIGHT_DECAY = 0.0001

START_WEIGHT_THRESHOLD = 3e-3
GAMMA = 0.99
TAU = 9e-3

START_NOISE = 0.9
END_NOISE = 0
NOISE_DECAY_STEPS = 175000


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGGER_CONFIG = {
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "tau": TAU,
    "actor_layer_1": ACTOR_LAYER_1,
    "actor_layer_2": ACTOR_LAYER_2,
    "critic_layer_1": CRITIC_LAYER_1,
    "critic_layer_2": CRITIC_LAYER_2,
    "actor_lr": ACTOR_LR,
    "critic_lr": CRITIC_LR,
    "actor_lr_step_size": ACTOR_LR_STEP_SIZE,
    "actor_lr_gamma": ACTOR_LR_GAMMA,
    "critic_lr_step_size": CRITIC_LR_STEP_SIZE,
    "critic_lr_gamma": CRITIC_LR_GAMMA,
    "start_noise": START_NOISE,
    "end_noise": END_NOISE,
    "noise_decay_steps": NOISE_DECAY_STEPS,
    "architecture": "DDPG",
    "current_actor_lr": ACTOR_LR,
    "current_critic_lr": CRITIC_LR,
    "actor_loss": 0,
    "critic_loss": 0,
    "avg_reward" : 0,
    "achieve_rate" : 0,
    "loss_rate" : 0,
    "anglevel_reward" : 0,
    "vel_reward" : 0
}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, ACTOR_LAYER_1)
        self.l2 = nn.Linear(ACTOR_LAYER_1, ACTOR_LAYER_2)
        self.l3 = nn.Linear(ACTOR_LAYER_2, action_dim)
        self.tanh = nn.Tanh()
        init.xavier_uniform_(self.l1.weight)
        init.xavier_uniform_(self.l2.weight)
        init.xavier_uniform_(self.l3.weight)

    def forward(self, state):
        state = state.to(torch.float32)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = torch.tanh(a)
        return a

    def forward_with_noise(self, state, noise, step):
        state = state.to(torch.float32)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = noise.get_action(a, step)
        a = torch.tanh(torch.FloatTensor(a).to(DEVICE))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, CRITIC_LAYER_1)
        self.l2 = nn.Linear(CRITIC_LAYER_1, CRITIC_LAYER_2)
        self.l3 = nn.Linear(CRITIC_LAYER_2, 1)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.l1.weight)
        init.xavier_uniform_(self.l2.weight)
        init.xavier_uniform_(self.l3.weight)

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
    def __init__(self, state_dim, action_dim, device, map, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device

        if config is None:
            self.config = None
            self.actor_lr = ACTOR_LR
            self.critic_lr = CRITIC_LR

            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.actor_target = Actor(state_dim, action_dim).to(self.device)
            self.actor_optim = optim.AdamW(self.actor.parameters(), lr=ACTOR_LR, weight_decay=ACTOR_LR_WEIGHT_DECAY)

            self.critic = Critic(state_dim, action_dim).to(self.device)
            self.critic_target = Critic(state_dim, action_dim).to(self.device)
            self.critic_optim = optim.AdamW(self.critic.parameters(), lr=CRITIC_LR, weight_decay=CRITIC_LR_WEIGHT_DECAY)

            self.hard_update(self.actor_target, self.actor)
            self.hard_update(self.critic_target, self.critic)

            self.mem = ReplayMemory(10000000)
            self.norm_mem = ReplayMemory(1)
            if SEPERATE_NORM_MEM:
                self.norm_mem = ReplayMemory(10000000)
            self.criterion = nn.MSELoss()

            self.normalizer = Normalizer(map)

            self.actor_lr_scheduler = StepLR(self.actor_optim, step_size=ACTOR_LR_STEP_SIZE, gamma=ACTOR_LR_GAMMA)
            self.critic_lr_scheduler = StepLR(self.critic_optim, step_size=CRITIC_LR_STEP_SIZE, gamma=CRITIC_LR_GAMMA)
        else:
            self.config = config
            self.actor_lr = config.actor_lr
            self.critic_lr = config.critic_lr

            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.actor_target = Actor(state_dim, action_dim).to(self.device)
            self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=config.actor_lr_weight_decay)

            self.critic = Critic(state_dim, action_dim).to(self.device)
            self.critic_target = Critic(state_dim, action_dim).to(self.device)
            self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=config.critic_lr_weight_decay)

            self.hard_update(self.actor_target, self.actor)
            self.hard_update(self.critic_target, self.critic)

            self.mem = ReplayMemory(10000000)
            self.norm_mem = ReplayMemory(1)
            if SEPERATE_NORM_MEM:
                self.norm_mem = ReplayMemory(10000000)
            self.criterion = nn.MSELoss()

            self.normalizer = Normalizer(map)

            self.actor_lr_scheduler = StepLR(self.actor_optim, step_size=config.actor_lr_step_size, gamma=config.actor_lr_gamma)
            self.critic_lr_scheduler = StepLR(self.critic_optim, step_size=config.critic_lr_step_size, gamma=config.critic_lr_gamma)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state):
        action = self.actor.forward(state)
        return action

    def get_action_with_noise(self, state, noise, step):
        action = self.actor.forward_with_noise(state, noise, step)
        return action
    
    def normalize_state(self, state):
        return self.normalizer.NormalizeState(state)

    def add_to_memory(self, state, action, next_state, reward):
        self.mem.push(state, action, next_state, reward)
        self.norm_mem.push(self.normalize_state(state), action, self.normalize_state(next_state), reward)

    def update_parameters(self, batch_size):
        if len(self.mem) < batch_size:
            return 0,0
        if SEPERATE_NORM_MEM:
            batch = Transition(*zip(*self.norm_mem.sample(batch_size)))
            states = torch.stack(batch.state).to(self.device).float()
            next_states = torch.stack(batch.next_state).to(self.device).float()
        else:
            batch = Transition(*zip(*self.mem.sample(batch_size)))
            states = torch.stack(self.normalizer.NormalizeStates(self.mem, batch.state)).to(self.device).float()
            next_states = torch.stack(self.normalizer.NormalizeStates(self.mem, batch.next_state)).to(self.device).float()
        actions = torch.stack(batch.action).to(self.device).float()
        rewards = torch.stack(batch.reward).to(self.device).float()
        #Critic loss
        QVal = self.critic((states, actions))
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target((next_states, next_actions.detach()))
        QPrime = rewards + GAMMA * next_Q
        critic_loss = self.criterion(QVal, QPrime)
        critic_loss = critic_loss.float()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        #Actor loss
        actor_losses = self.critic.forward((states, self.actor.forward(states)))
        actor_loss = -actor_losses.mean()
        #Update networks
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.config is None:
            self.soft_update(self.actor_target, self.actor, TAU)
            self.soft_update(self.critic_target, self.critic, TAU)
        else:
            self.soft_update(self.actor_target, self.actor, self.config.tau)
            self.soft_update(self.critic_target, self.critic, self.config.tau)

        clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        clip_grad_norm_(self.critic.parameters(), max_norm=1.0)


        self.critic_lr_scheduler.step()
        self.actor_lr_scheduler.step()
        return critic_loss.item(), actor_loss.item()
    def evaluate(self, venv, reward_func, eval_episodes=10):
        avg_reward = 0
        achieve_rate = 0
        for i in range(eval_episodes):
            state, startdist = venv.reset()
            state = torch.FloatTensor(state).to(self.device)
            ep_reward = 0
            for step in range(100):
                action = self.actor(state).cpu().data.numpy().flatten()
                next_state, collision, done, achieved_goal, dist_traveled = venv.step(action, step)
                reward, tw, dw, aw = reward_func(done, collision, achieved_goal, action[0], action[1],
                                                     next_state[5])
                ep_reward += reward
                state = torch.FloatTensor(next_state).to(self.device)
                if done:
                    break
                if achieved_goal:
                    achieve_rate += 1
            avg_reward += ep_reward
        avg_reward = avg_reward / eval_episodes
        return avg_reward, achieve_rate*10


class TrainingExecutor:
    def __init__(self, map):
        self.ddpg = DDPG(STATE_DIM, ACTION_DIM, DEVICE, map)
        self.plotter = None
        self.logger = None

    def get_reward(self, done, collision, achieved_goal, anglevel, vel, min_dist, config=None):
        '''
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
        return total_weight, time_weight, dist_weight, angle_weight'''
        d = lambda x: 1-x if x<1 else 0
        if done:
            if achieved_goal:
                return FINISH_WEIGHT, 1, 1, 1
            else:
                return COLLISION_WEIGHT/2, 1, 1, 1
        if collision:
            return COLLISION_WEIGHT, 1, 1, 1
        return (vel)*SPEED_WEIGHT+abs(anglevel)*ANGLE_SPEED_WEIGHT+d(min_dist)*MIN_DIST_WEIGHT+TIME_WEIGHT, ((vel)*SPEED_WEIGHT).item(), (abs(anglevel)*ANGLE_SPEED_WEIGHT).item(), d(min_dist)*MIN_DIST_WEIGHT


    def train(self, env, num_episodes=EPISODES, max_steps=MAX_TIMESTEP, batch_size=BATCH_SIZE, start_episode=0, config=None, map=None, plotter_display=True):
        if config is None:
            noise = OUNoise(ACTION_DIM, max_sigma=START_NOISE, min_sigma=END_NOISE, decay_period=NOISE_DECAY_STEPS)
            rewards = []
            self.logger = wandb.init(project="autonav", config=LOGGER_CONFIG, name="cuda-v1 test commit")
            if self.plotter is None:
                self.plotter = Model_Plotter(num_episodes, plotter_display)
            visualizer = None
            if VISUALIZER_ENABLED:
                visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)

            total_steps = 0
            for episode in range(start_episode, num_episodes):
                state, dist = env.reset()
                if VISUALIZER_ENABLED:
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
                episode_closs = []
                episode_aloss = []
                action_q = []
                ovr_dist = 0

                states = []
                actions = []

                for step in range(max_steps):
                    nstate = self.ddpg.normalize_state(state).to(DEVICE)
                    action = self.ddpg.get_action_with_noise(nstate, noise, total_steps)
                    actions.append(action)
                    states.append(state)
                    next_state, collision, done, achieved_goal, dist_traveled = env.step(action, step)
                    if step == max_steps-1:
                        done = True
                    ovr_dist += dist_traveled
                    reward, tw, dw, aw = self.get_reward(done, collision, achieved_goal, action[0], action[1], next_state[5])
                    self.ddpg.add_to_memory(state, action.to(DEVICE), torch.tensor(next_state).to(DEVICE), torch.tensor([reward]).to(DEVICE))
                    episode_reward += reward
                    episode_tw += tw
                    episode_dw += dw
                    episode_aw += aw
                    episode_x.append(next_state[1])
                    episode_y.append(next_state[2])
                    state = torch.FloatTensor(next_state).to(DEVICE)

                    c_loss, a_loss = self.ddpg.update_parameters(batch_size)
                    episode_closs.append(c_loss)
                    episode_aloss.append(a_loss)

                    total_steps += 1
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
                if VISUALIZER_ENABLED:
                    visualizer.clear()
                    visualizer.update(episode_x, episode_y, action_q)
                if SAVE_FREQ != -1 and episode % SAVE_FREQ == 0:
                    self.save(episode)
                if EVAL_FREQ != -1 and (episode+1) % EVAL_FREQ == 0:
                    eval_rew, eval_ac = self.ddpg.evaluate(env, self.get_reward)
                else:
                    eval_rew = -1
                    eval_ac = -1

                rewards.append(episode_reward)
                print("Episode: " + str(episode) + " Reward: " + str(episode_reward))
                self.plotter.update(episode, initdist, episode_reward, episode_dw, episode_aw, episode_tw, episode_achieve, episode_collide, sum(episode_closs)/len(episode_closs), sum(episode_aloss)/len(episode_aloss), eval_rew, eval_ac)
        else:
            noise = OUNoise(ACTION_DIM, max_sigma=config.start_noise, min_sigma=END_NOISE, decay_period=config.noise_decay_steps)
            rewards = []
            if self.plotter is None:
                self.plotter = Model_Plotter(num_episodes, plotter_display)
            visualizer = None
            if VISUALIZER_ENABLED:
                visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)
            self.ddpg = DDPG(STATE_DIM, ACTION_DIM, DEVICE, map, config=config)
            total_steps = 0
            for episode in range(start_episode, num_episodes):
                state, dist = env.reset()
                if VISUALIZER_ENABLED:
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
                episode_closs = []
                episode_aloss = []
                action_q = []
                ovr_dist = 0

                states = []
                actions = []

                for step in range(max_steps):
                    nstate = self.ddpg.normalize_state(state).to(DEVICE)
                    action = self.ddpg.get_action_with_noise(nstate, noise, total_steps)
                    actions.append(action)
                    states.append(state)
                    next_state, collision, done, achieved_goal, dist_traveled = env.step(action, step)
                    if step == max_steps - 1:
                        done = True
                    ovr_dist += dist_traveled
                    reward, tw, dw, aw = self.get_reward(done, collision, achieved_goal, action[0], action[1],
                                                         next_state[5])
                    self.ddpg.add_to_memory(state, action.to(DEVICE), torch.tensor(next_state).to(DEVICE),
                                            torch.tensor([reward]).to(DEVICE))
                    episode_reward += reward
                    episode_tw += tw
                    episode_dw += dw
                    episode_aw += aw
                    episode_x.append(next_state[1])
                    episode_y.append(next_state[2])
                    state = torch.FloatTensor(next_state).to(DEVICE)

                    c_loss, a_loss = self.ddpg.update_parameters(BATCH_SIZE)
                    episode_closs.append(c_loss)
                    episode_aloss.append(a_loss)

                    total_steps += 1
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
                if VISUALIZER_ENABLED:
                    visualizer.clear()
                    visualizer.update(episode_x, episode_y, action_q)
                if SAVE_FREQ != -1 and episode % SAVE_FREQ == 0:
                    self.save(episode)
                if EVAL_FREQ != -1 and (episode+1) % EVAL_FREQ == 0:
                    eval_rew, eval_ac = self.ddpg.evaluate(env, self.get_reward)
                else:
                    eval_rew = -1
                    eval_ac = -1

                rewards.append(episode_reward)
                print("Episode: " + str(episode) + " Reward: " + str(episode_reward))
                self.plotter.update(episode, initdist, episode_reward, episode_dw, episode_aw, episode_tw,
                                    episode_achieve, episode_collide, sum(episode_closs) / len(episode_closs),
                                    sum(episode_aloss) / len(episode_aloss), eval_rew, eval_ac)
        wandb.finish()

    def test(self, env, num_episodes=EPISODES, max_steps=MAX_TIMESTEP, batch_size=BATCH_SIZE, start_episode=0):
        self.ddpg.actor.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_actor.pth"))
        self.ddpg.critic.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_critic.pth"))
        rewards = []
        visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)
        for episode in range(start_episode, num_episodes):
            state, dist = env.reset()
            visualizer.start(state[1], state[2], state[3], state[4])
            initdist = dist
            initangle = state[1]
            state = torch.FloatTensor(state).to(DEVICE)
            episode_reward = 0
            episode_tw = 0
            episode_dw = 0
            episode_aw = 0
            episode_achieve = 0
            episode_collide = 0
            episode_x = []
            episode_y = []
            episode_closs = []
            episode_aloss = []
            action_q = []
            ovr_dist = 0
            states = []
            actions = []
            for step in range(max_steps):
                nstate = self.ddpg.normalize_state(state).to(DEVICE)
                action = self.ddpg.get_action(nstate)
                actions.append(action)
                states.append(state)
                next_state, collision, done, achieved_goal, dist_traveled = env.step(action, step)
                ovr_dist += dist_traveled
                reward, tw, dw, aw = self.get_reward(done, collision, achieved_goal, action[0], action[1], next_state[5])
                episode_reward += reward
                episode_tw += tw
                episode_dw += dw
                episode_aw += aw
                episode_x.append(next_state[1])
                episode_y.append(next_state[2])
                state = torch.FloatTensor(next_state).to(DEVICE)
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
            rewards.append(episode_reward)
            print("Episode: " + str(episode) + " Reward: " + str(episode_reward))



    def save(self, episode, filename=FILE_NAME, directory=FILE_LOCATION):
        if not os.path.exists(directory + "/" + filename):
            os.mkdir(directory + "/" + filename)
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
                "none_chance_y": np.asarray(stats["none_chance_y"]),
                "critic_loss_y": np.asarray(stats["critic_loss_y"]),
                "actor_loss_y": np.asarray(stats["actor_loss_y"]),
            }
            self.plotter.load(np_stats)
        with open(f"{directory}/{filename}/memory.json", "r") as infile:
            self.ddpg.mem.from_json(infile.read(), DEVICE)
        with open(f"{directory}/{filename}/core.json", "r") as infile:
            episode = json.load(infile)["episode"]
        self.ddpg.actor.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.ddpg.critic.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic.pth"))
        return episode
