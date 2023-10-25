import math

import torch
import numpy as np
import torch.nn as nn
import os
import sys
import json
import wandb
import keyboard
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from torch.nn import init
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from drl_utils import ReplayMemory, Transition, NumpyArrayEncoder, Model_Plotter, OUNoise, Model_Visualizer, Normalizer, Minima_Visualizer, EGreedyNoise, ProgressiveRewards
from logger import logger
import torch.optim as optim

SWEEP_CONFIG = {

}

if sys.platform == "win32":
    FILE_LOCATION = "G:/Projects/AutoNav/AutoNavServer/assets/drl/models"
elif sys.platform == "linux" or sys.platform == "linux2":
    FILE_LOCATION = "/home/jovyan/workspace/AutoNavServer/assets/drl/models"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
FILE_NAME = "SampleModel"
SAVE_FREQ = 999
EVAL_FREQ = -1
POLICY_FREQ = 12
VISUALIZER_ENABLED = False
OPTIMIZE = True

DEBUG_SAME_SITUATION = False
DEBUG_CIRCLE = False
DEBUG_CRITIC = False

#EPISODES = 30000
TOTAL_TIMESTEPS = 10000000
MAX_TIMESTEP = 100
BATCH_SIZE = 50

COLLISION_WEIGHT = -15
NONE_WEIGHT = 0
TIME_WEIGHT = -0.075#-6
FINISH_WEIGHT = 10
DIST_WEIGHT = 0.025
PASS_DIST_WEIGHT = 0
CHALLENGE_WEIGHT = 0.01
CHALLENGE_EXP_BASE = 1
ANGLE_WEIGHT = 0.25#-2
ANGLE_THRESH = 0.75
SPEED_WEIGHT = -0.5
ANGLE_SPEED_WEIGHT = -0.5#-0.5
MIN_DIST_WEIGHT = 0
WALL_DIST = 0
ANGLE_DECAY = 1
CLOSER_WEIGHT = 1
CLOSER_O_WEIGHT = -1
hdg_function = lambda x: 1/(ANGLE_THRESH*np.sqrt(2*np.pi)) * math.exp(-(x ** 2 / (2 * ANGLE_THRESH) ** 2))
hdg_decay_function = lambda x: ANGLE_DECAY**(ANGLE_THRESH*x)

STATE_DIM = 8
ACTION_DIM = 2

ACTOR_LAYER_1 = 128
ACTOR_LAYER_2 = 64

ACTOR_LR = 1e-3
ACTOR_LR_STEP_SIZE = 5e6
ACTOR_LR_GAMMA = 0.1
ACTOR_LR_WEIGHT_DECAY = 0.0001

CRITIC_LAYER_1 = 256
CRITIC_LAYER_2 = 128

CRITIC_LR = 1e-6
CRITIC_LR_STEP_SIZE = 5e6
CRITIC_LR_GAMMA = 0.1
CRITIC_LR_WEIGHT_DECAY = 0.0001

START_WEIGHT_THRESHOLD = 3e-3
GAMMA = 0.9999
TAU = 0.005

START_NOISE = 1
END_NOISE = 0.1
NOISE_DECAY_STEPS = 200000


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
        a = f.relu(self.l1(state))
        a = f.relu(self.l2(a))
        a = self.l3(a)
        a = torch.tanh(a)
        return a

    def forward_with_noise(self, state, noise, step):
        state = state.to(torch.float32)
        a = f.relu(self.l1(state))
        a = f.relu(self.l2(a))
        a = self.l3(a)
        a = torch.tanh(a)
        a = noise.get_action(a, step)
        a = torch.tanh(torch.FloatTensor(a).to(DEVICE))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, CRITIC_LAYER_1)
        self.l2_s = nn.Linear(CRITIC_LAYER_1, CRITIC_LAYER_2)
        self.l2_a = nn.Linear(action_dim, CRITIC_LAYER_2)
        self.l3 = nn.Linear(CRITIC_LAYER_2, 1)

        self.l4 = nn.Linear(state_dim, CRITIC_LAYER_1)
        self.l5_s = nn.Linear(CRITIC_LAYER_1, CRITIC_LAYER_2)
        self.l5_a = nn.Linear(action_dim, CRITIC_LAYER_2)
        self.l6 = nn.Linear(CRITIC_LAYER_2, 1)
        self.relu = nn.ReLU()
        init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l2_s.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l2_a.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l3.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l4.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l5_s.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l5_a.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.l6.weight, nonlinearity='relu')


    def forward(self, xs):
        x, a = xs
        x = x.to(torch.float32)
        a = a.to(torch.float32)
        s1= self.relu(self.l1(x))
        self.l2_s(s1)
        self.l2_a(a)
        s11 = torch.mm(s1, self.l2_s.weight.t())
        s12 = torch.mm(a, self.l2_a.weight.t())
        s1 = self.relu(s11+s12 + self.l2_a.bias.data)
        q1 = self.l3(s1)

        s2 = self.relu(self.l4(x))
        self.l5_s(s2)
        self.l5_a(a)
        s21 = torch.mm(s2, self.l5_s.weight.t())
        s22 = torch.mm(a, self.l5_a.weight.t())
        s2 = self.relu(s21+s22 + self.l5_a.bias.data)
        q2 = self.l6(s2)
        """
        q1 = self.relu(self.l1(torch.cat([x,a], 1)))
        q1 = self.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = self.relu(self.l4(torch.cat([x,a], 1)))
        q2 = self.relu(self.l5(q2))
        q2 = self.l6(q2)"""
        return q1, q2
    def q1(self, xs):
        x, a = xs
        x = x.to(torch.float32)
        a = a.to(torch.float32)
        s1= self.relu(self.l1(x))
        self.l2_s(s1)
        self.l2_a(a)
        s11 = torch.mm(s1, self.l2_s.weight.t())
        s12 = torch.mm(a, self.l2_a.weight.t())
        s1 = self.relu(s11+s12 + self.l2_a.bias.data)
        q1 = self.l3(s1)
        return q1

class TD3(object):
    def __init__(self, state_dim, action_dim, device, inpmap, config=None, policy_freq=POLICY_FREQ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_freq = policy_freq
        self.actor_loss = 0

        self.device = device

        self.update_steps = 0

        if config is None:
            self.config = None
            self.actor_lr = ACTOR_LR
            self.critic_lr = CRITIC_LR
            actor_lr_gamma = ACTOR_LR_GAMMA
            critic_lr_gamma = CRITIC_LR_GAMMA
            actor_lr_step_size = ACTOR_LR_STEP_SIZE
            critic_lr_step_size = CRITIC_LR_STEP_SIZE
        else:
            self.config = config
            self.actor_lr = config.actor_lr
            self.critic_lr = config.critic_lr
            actor_lr_gamma = config.actor_lr_gamma
            critic_lr_gamma = config.critic_lr_gamma
            actor_lr_step_size = config.actor_lr_step_size
            critic_lr_step_size = config.critic_lr_step_size

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.norm_mem = ReplayMemory(10000000)
        self.criterion = f.huber_loss

        self.normalizer = Normalizer(inpmap)

        #self.actor_lr_scheduler = StepLR(self.actor_optim, step_size=config.actor_lr_step_size, gamma=config.actor_lr_gamma)
        #self.critic_lr_scheduler = StepLR(self.critic_optim, step_size=config.critic_lr_step_size, gamma=config.critic_lr_gamma)
        self.actor_lr_scheduler = ReduceLROnPlateau(self.actor_optim, "max", threshold=10, threshold_mode="abs", patience=5, factor=actor_lr_gamma)
        self.critic_lr_scheduler = ReduceLROnPlateau(self.critic_optim, "max", threshold=10, threshold_mode="abs", patience=5, factor=critic_lr_gamma)

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state):
        action = self.actor.forward(state)
        return action

    def get_action_with_noise(self, state, noise, step):
        action = self.actor.forward_with_noise(state, noise, step)
        return action#torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    
    def normalize_state(self, state):
        return self.normalizer.NormalizeState(state)

    def add_to_memory(self, state, action, next_state, reward, done):
        d = lambda x: 0 if x is True else 1
        self.norm_mem.push(self.normalize_state(state), action, self.normalize_state(next_state), reward, torch.FloatTensor([d(done)]))

    def update_parameters(self, iters, batch_size, noise, step, achieve_chance, mem=None):
        c_loss = []
        a_loss = []
        if mem is None: mem = self.norm_mem
        for i in range(iters):
            if len(mem) < batch_size:
                return 0,0
            batch = Transition(*zip(*mem.sample(batch_size)))
            states = torch.stack(batch.state).to(self.device).float()
            next_states = torch.stack(batch.next_state).to(self.device).float()
            actions = torch.stack(batch.action).to(self.device).float()
            rewards = torch.stack(batch.reward).to(self.device).float()
            dones = torch.stack(batch.done).to(self.device).float()
            with torch.no_grad():
                noise = torch.Tensor(actions).data.normal_(0, 0.2).to(self.device)
                noise = noise.clamp(-0.5, 0.5)
                next_actions = self.actor_target.forward(states)
                next_actions = torch.clamp(next_actions+noise, -1, 1)
            #Critic loss
            QVal, QVal2 = self.critic((states, actions))
            next_Q1, next_Q2 = self.critic_target((next_states, next_actions.detach()))
            next_Q_min = torch.min(next_Q1, next_Q2)
            QPrime = rewards + dones * GAMMA * next_Q_min
            QPrime = QPrime.detach()

            closs1 = self.criterion(QVal, QPrime).float()
            closs2 = self.criterion(QVal2, QPrime).float()

            critic_loss = closs1+closs2

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            #Update networks
            if self.update_steps % self.policy_freq == 0:
                # Actor loss
                actor_loss = self.critic.q1((states, self.actor.forward(states)))
                self.actor_loss = -actor_loss.mean()
                self.actor_optim.zero_grad()
                self.actor_loss.backward()
                self.actor_optim.step()
                clip_grad_norm_(self.actor.parameters(), max_norm=10000)

                if self.config is None:
                    self.soft_update(self.actor_target, self.actor, TAU)
                else:
                    self.soft_update(self.actor_target, self.actor, self.config.tau)
            if self.config is None:
                self.soft_update(self.critic_target, self.critic, TAU)
            else:
                self.soft_update(self.critic_target, self.critic, self.config.tau)
            clip_grad_norm_(self.critic.parameters(), max_norm=10000)


            #self.critic_lr_scheduler.step()
            #self.actor_lr_scheduler.step()
            #self.critic_lr_scheduler.step(achieve_chance)
            #self.actor_lr_scheduler.step(achieve_chance)
            self.update_steps += 1
            c_loss.append(critic_loss.item())
            a_loss.append(self.actor_loss.item())
        return sum(c_loss)/len(c_loss), sum(a_loss)/len(a_loss)
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
    def __init__(self, inpmap, logger_path, config=None):
        self.network = TD3(STATE_DIM, ACTION_DIM, DEVICE, inpmap, config)
        self.plotter = None
        self.logger = None
        self.sys_logs = logger("training.log", logger_path)



    @staticmethod
    def get_reward(done, collision, achieved_goal, anglevel, vel, min_dist, angle_error, timesteps):
        d = lambda x: 1-x if x<1 else 0

        if done:
            if achieved_goal:
                return FINISH_WEIGHT, 1, 1, 1, 1
            elif collision:
                return COLLISION_WEIGHT, 1, 1, 1, 1
            else:
                return NONE_WEIGHT, 1, 1, 1, 1
        hdg_reward = hdg_function(angle_error/np.pi)*ANGLE_WEIGHT*hdg_decay_function(timesteps)
        return abs(1-vel)*SPEED_WEIGHT+abs(anglevel)*ANGLE_SPEED_WEIGHT+d(min_dist)*MIN_DIST_WEIGHT+TIME_WEIGHT+hdg_reward, (abs(1-vel)*SPEED_WEIGHT).item(), (abs(anglevel)*ANGLE_SPEED_WEIGHT).item(), TIME_WEIGHT, hdg_reward#d(min_dist)*MIN_DIST_WEIGHT

    @staticmethod
    def get_reward_beta(done, collision, achieved_goal, goaldist, angle_error):
        hdg_function = lambda x: 1 if abs(x)<np.pi/8 else 0
        if done:
            if achieved_goal:
                return FINISH_WEIGHT, 1, 1, 1
            elif collision:
                return COLLISION_WEIGHT, 1, 1, 1
            else:
                return NONE_WEIGHT, 1, 1, 1
        goal_reward = (1.0 - min(1, goaldist/10))*DIST_WEIGHT
        hdg_reward = hdg_function(angle_error)*((np.pi/8-angle_error)*8/np.pi)*ANGLE_WEIGHT
        return goal_reward+hdg_reward+TIME_WEIGHT, hdg_reward, goal_reward, 0


    def train(self, env, total_ts=TOTAL_TIMESTEPS, max_steps=MAX_TIMESTEP, batch_size=BATCH_SIZE, start_ts=0, config=None, inpmap=None, plotter_display=True, test=False):
        self.sys_logs.logs(["Training session started"
                            ,"Training details:"
                            ,"Total timesteps: " + str(total_ts)
                            ,"Max steps: " + str(max_steps)
                            ,"Batch size: " + str(batch_size)
                            ,"Start timestep: " + str(start_ts)
                            ,"Config: " + str(config)
                            ,"Is test: " + str(test)])
        if test:
            self.sys_logs.log("Attempting to load model")
            try:
                self.network.actor.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_actor.pth"))
                self.network.actor_target.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_actor.pth"))
                self.network.critic.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_critic.pth"))
                self.network.critic_target.load_state_dict(torch.load(f"{FILE_LOCATION}/{FILE_NAME}/agent_critic.pth"))
            except Exception as e:
                self.sys_logs.log("Failed to load model. Exception: " + str(e), logtype="e")
                return
        else:
            self.logger = wandb.init(project="autonav", config=LOGGER_CONFIG, name="cuda-v1 test commit")
            wandb.watch(self.network.actor, log='all', log_freq=10)
            wandb.watch(self.network.critic, log='all', log_freq=10)
        if config is None:
            start_noise = START_NOISE
            noise_decay_steps = NOISE_DECAY_STEPS
            time_weight = TIME_WEIGHT
            angle_speed_weight = ANGLE_SPEED_WEIGHT
            closer_weight = CLOSER_WEIGHT
            closer_o_weight = CLOSER_O_WEIGHT
        else:
            start_noise = config.start_noise
            noise_decay_steps = config.noise_decay_steps
            time_weight = config.time_weight
            angle_speed_weight = config.angle_speed_weight
            closer_weight = config.closer_weight
            closer_o_weight = config.closer_o_weight
        #noise = OUNoise(ACTION_DIM, max_sigma=start_noise, min_sigma=END_NOISE, decay_period=noise_decay_steps)
        noise = EGreedyNoise(ACTION_DIM, max_sigma=start_noise, min_sigma=END_NOISE, decay_period=noise_decay_steps)
        if self.plotter is None and not test:
            self.plotter = Model_Plotter(total_ts, plotter_display, self.network.norm_mem)
        visualizer = None
        if VISUALIZER_ENABLED or test:
            visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)
            keyboard.add_hotkey("ctrl+q", lambda: visualizer.toggle())
        rewardfunc = ProgressiveRewards(ANGLE_THRESH, ANGLE_DECAY, SPEED_WEIGHT, angle_speed_weight, ANGLE_WEIGHT, time_weight, closer_weight, closer_o_weight)
        circle = False
        pre_rewards = []

        if not DEBUG_CIRCLE:
            pretrain_mem = ReplayMemory(10000)
            total = 0
            doneconversion = lambda x: 0 if x is True else 1
            while circle is False:
                state, distance, min_dist, circle = env.debug_circle_reset()
                state = torch.FloatTensor(state).to(self.network.device)
                if circle is True:
                    break
                for ts in range(100):
                    action = torch.FloatTensor([0, 1]).to(self.network.device)
                    next_state, collision, done, achieved_goal, dist_traveled, min_dist = env.step(action)
                    reward, vw, avw, tw, aw, caw, cw = rewardfunc.get_reward(next_state[1], min_dist, next_state[0], action[0],
                                                                    action[1], ts)
                    pretrain_mem.push(state, action, torch.FloatTensor(next_state).to(self.network.device), torch.FloatTensor([reward.item()]).to(self.network.device), torch.FloatTensor([doneconversion(done)]).to(self.network.device))
                    total += 1
                    state = torch.FloatTensor(next_state).to(self.network.device)
                    if done:
                        break
            self.network.update_parameters(total, total, noise, 0, 0, mem=pretrain_mem)
        if DEBUG_CIRCLE:
            circle_visualizer = Minima_Visualizer(env.basis.size.width, env.basis.size.height)
            state, initdist, min_dist, circle_visualizer.shouldshow = env.debug_circle_reset()
        else:
            state, initdist, min_dist = env.reset(reload=DEBUG_SAME_SITUATION)
        state = torch.FloatTensor(state).to(DEVICE)
        noise.reset()
        episode_reward = 0
        episode_vw = 0
        episode_avw = 0
        episode_tw = 0
        episode_aw = 0
        episode_cw = 0
        episode_caw = 0
        episode_achieve = 0
        episode_collide = 0
        episode_x = []
        episode_y = []
        ovr_dist = 0

        end_rewards = []

        states = []
        actions = []
        rewards = []
        total_states = []
        total_actions = []
        ep_steps = 0
        eps = 0
        done = False
        for timestep in range(start_ts, total_ts):
            nstate = self.network.normalize_state(state).to(DEVICE)
            if not DEBUG_CIRCLE:
                action = self.network.get_action_with_noise(nstate, noise, timestep)
            else:
                action = torch.FloatTensor([0,1]).to(self.network.device)
            actions.append(action)
            states.append(state)
            if DEBUG_CIRCLE:
                total_states.append(state)
                total_actions.append(action)
            a_in = action
            a_in[1] = (a_in[1]+1)/2
            if not OPTIMIZE:
                a_in = torch.FloatTensor([0,1]).to(self.network.device)
            next_state, collision, done, achieved_goal, dist_traveled, min_dist = env.step(a_in)
            if ep_steps >= max_steps-1:
                done = True
            ovr_dist += dist_traveled
            reward, vw, avw, tw, aw, caw, cw = rewardfunc.get_reward(next_state[1], min_dist, next_state[0], action[0], action[1], ep_steps)
            #reward, vw, avw, aw = self.get_reward_beta(done, collision, achieved_goal, next_state[1], next_state[0])
            rewards.append(reward)
            self.network.add_to_memory(state, action.to(DEVICE), torch.tensor(next_state).to(DEVICE), torch.tensor([reward]).to(DEVICE), done)
            episode_reward += reward
            episode_vw += vw
            episode_avw += avw
            episode_tw += tw
            episode_aw += aw
            episode_cw += cw
            episode_caw += caw
            episode_x.append(next_state[2])
            episode_y.append(next_state[3])
            state = torch.FloatTensor(next_state).to(DEVICE)
            #visualizer.update(episode_x[ep_steps], episode_y[ep_steps], self.network.critic.forward((states[ep_steps], actions[ep_steps])))
            ep_steps+=1
            if collision is True:
                episode_collide = 1
                done = True
            if achieved_goal is True:
                episode_achieve = 1
            if done:
                end_rewards.append([episode_reward, episode_vw, episode_avw, episode_tw, episode_aw])
                if not test and not DEBUG_CIRCLE:
                    c_loss, a_loss = 0, 0
                    if timestep != 0 and OPTIMIZE:
                        c_loss, a_loss = self.network.update_parameters(ep_steps, batch_size, noise, timestep,
                                                                        self.plotter.get_achieve_chance(timestep))
                    if SAVE_FREQ != -1 and (eps+1) % SAVE_FREQ == 0:
                        self.save_model()
                        #self.save(timestep)
                    if EVAL_FREQ != -1 and (eps+1) % EVAL_FREQ == 0:
                        eval_rew, eval_ac = self.network.evaluate(env, self.get_reward)
                    else:
                        eval_rew = -1
                        eval_ac = -1
                    self.plotter.update(eps, initdist, episode_reward, episode_vw, episode_avw, episode_tw,
                                        episode_achieve, episode_collide, c_loss,
                                        a_loss, eval_rew, eval_ac, episode_caw, episode_cw)
                print("Episode: " + str(eps) + " Reward: " + str(episode_reward))
                if DEBUG_CIRCLE:
                    state, initdist, min_dist, circle_visualizer.shouldshow = env.debug_circle_reset()
                    if circle_visualizer.shouldshow is True:
                        if DEBUG_CRITIC:
                            total_s = []
                            for i in range(len(total_states)):
                                total_s.append(self.network.normalize_state(total_states[i]).to(DEVICE))
                            rews = self.network.critic.q1((torch.cat(total_s).to(self.network.device), torch.cat(total_actions).to(self.network.device)))
                            circle_visualizer.generate(total_states, torch.cat(rews).cpu().detach().numpy(), end_rewards)
                        else:
                            circle_visualizer.generate(total_states, rewards, end_rewards)
                        break
                else:
                    state, initdist, min_dist = env.reset(reload=DEBUG_SAME_SITUATION)
                if (VISUALIZER_ENABLED or test) and len(states) > 0 and visualizer.show:
                    states = torch.stack(states).to(self.network.device)
                    actions = torch.stack(actions).to(self.network.device)
                    action_q = self.network.critic.forward((states, actions))
                    visualizer.clear()
                    visualizer.update(episode_x, episode_y, action_q)
                    visualizer.start(state[2], state[3], state[4], state[5])
                state = torch.FloatTensor(state).to(DEVICE)
                noise.reset()
                episode_reward = 0
                episode_vw = 0
                episode_avw = 0
                episode_tw = 0
                episode_aw = 0
                episode_cw = 0
                episode_caw = 0
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
                ep_steps = 0
                eps += 1
                done=False
                rewardfunc.reset()
        if not test:
            wandb.finish()

    def save_model(self, filename=FILE_NAME, directory=FILE_LOCATION):
        if not os.path.exists(directory + "/" + filename):
            os.mkdir(directory + "/" + filename)
        torch.save(self.network.actor.state_dict(), f"{directory}/{filename}/agent_actor.pth")
        torch.save(self.network.critic.state_dict(), f"{directory}/{filename}/agent_critic.pth")

    def save(self, timestep, filename=FILE_NAME, directory=FILE_LOCATION):
        self.save_model()
        if self.plotter is not None:
            statistics = self.plotter.save()
        else:
            statistics = {}
        with open(f"{directory}/{filename}/statistics.json", "w") as outfile:
            json.dump(statistics, outfile, cls=NumpyArrayEncoder)
        with open(f"{directory}/{filename}/memory.json", "w") as outfile:
            outfile.write(self.network.norm_mem.to_json())
        with open(f"{directory}/{filename}/core.json", "w") as outfile:
            json.dump({
                "episode": timestep
            }, outfile)

    def load(self, timesteps=TOTAL_TIMESTEPS, filename=FILE_NAME, directory=FILE_LOCATION):
        with open(f"{directory}/{filename}/statistics.json", "r") as infile:
            stats = json.load(infile)
            self.plotter = Model_Plotter(timesteps)
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
                "critic_loss_y": np.asarray(stats["critic_loss"]),
                "actor_loss_y": np.asarray(stats["actor_loss"]),
            }
            self.plotter.load(np_stats)
        with open(f"{directory}/{filename}/memory.json", "r") as infile:
            self.network.norm_mem.from_json(infile.read(), DEVICE)
        with open(f"{directory}/{filename}/core.json", "r") as infile:
            episode = json.load(infile)["episode"]
        self.network.actor.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.network.critic.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic.pth"))
        self.network.actor_target.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.network.critic_target.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic.pth"))
        return episode
