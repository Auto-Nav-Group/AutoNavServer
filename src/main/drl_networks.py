import torch
import numpy as np
import torch.nn as nn
import os
import sys
import json
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.distributions import Normal
from drl_utils import ReplayMemory, Transition, NumpyArrayEncoder, Model_Plotter, Normalizer
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim

if sys.platform == "win32":
    FILE_LOCATION = "G:/Projects/AutoNav/AutoNavServer/assets/drl/models"
elif sys.platform == "linux" or sys.platform == "linux2":
    FILE_LOCATION = "/home/jovyan/workspace/AutoNavServer/assets/drl/models"
elif sys.platform == "darwin":
    FILE_LOCATION = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/drl/models"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
FILE_NAME = "SampleModel"
# EPISODES = 30000


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTOR_LAYER_1 = 128
ACTOR_LAYER_2 = 64

ACTOR_LR = 0.0007
ACTOR_LR_STEP_SIZE = 950000
ACTOR_LR_GAMMA = 0.1
ACTOR_LR_WEIGHT_DECAY = 0.0001

CRITIC_LAYER_1 = 256
CRITIC_LAYER_2 = 128

CRITIC_LR = 0.0001
CRITIC_LR_STEP_SIZE = 4000000
CRITIC_LR_GAMMA = 0.1
CRITIC_LR_WEIGHT_DECAY = 0.0001

POLICY_LAYER_1 = 128

START_WEIGHT_THRESHOLD = 3e-3
GAMMA = 0.9999
TAU = 0.005



class TD3(object):
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(TD3.Actor, self).__init__()
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

        def forward_with_noise(self, state, noise):
            state = state.to(torch.float32)
            a = f.relu(self.l1(state))
            a = f.relu(self.l2(a))
            a = self.l3(a)
            a = torch.tanh(a)
            a = noise.get_action(a)
            a = torch.tanh(torch.FloatTensor(a, device=DEVICE))
            return a

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(TD3.Critic, self).__init__()
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

        @staticmethod
        def unpack(xs):
            x, a = xs
            x = x.to(torch.float32)
            a = a.to(torch.float32)
            return x, a

        def forward(self, xs):
            x, a = TD3.Critic.unpack(xs)
            q1 = self.q1(xs)
            s2 = self.relu(self.l4(x))
            self.l5_s(s2)
            self.l5_a(a)
            s21 = torch.mm(s2, self.l5_s.weight.t())
            s22 = torch.mm(a, self.l5_a.weight.t())
            s2 = self.relu(s21 + s22 + self.l5_a.bias.data)
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
            x, a = TD3.Critic.unpack(xs)
            s1 = self.relu(self.l1(x))
            self.l2_s(s1)
            self.l2_a(a)
            s11 = torch.mm(s1, self.l2_s.weight.t())
            s12 = torch.mm(a, self.l2_a.weight.t())
            s1 = self.relu(s11 + s12 + self.l2_a.bias.data)
            q1 = self.l3(s1)
            return q1

    def __init__(self, state_dim, action_dim, device, inpmap, batch_size, config=None, policy_freq=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_freq = policy_freq
        self.batch_size = batch_size
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

        self.actor = TD3.Actor(state_dim, action_dim).to(self.device)
        self.actor_target = TD3.Actor(state_dim, action_dim).to(self.device)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

        self.critic = TD3.Critic(state_dim, action_dim).to(self.device)
        self.critic_target = TD3.Critic(state_dim, action_dim).to(self.device)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.mem = ReplayMemory(10000000)
        self.criterion = f.huber_loss

        self.normalizer = Normalizer(inpmap)

        # self.actor_lr_scheduler = StepLR(self.actor_optim, step_size=config.actor_lr_step_size, gamma=config.actor_lr_gamma)
        # self.critic_lr_scheduler = StepLR(self.critic_optim, step_size=config.critic_lr_step_size, gamma=config.critic_lr_gamma)
        self.actor_lr_scheduler = StepLR(self.actor_optim, step_size=actor_lr_step_size, gamma=actor_lr_gamma)#ReduceLROnPlateau(self.actor_optim, "max", threshold=10, threshold_mode="abs",

                                                    #patience=5, factor=actor_lr_gamma)

        self.critic_lr_scheduler = StepLR(self.critic_optim, step_size=critic_lr_step_size, gamma=critic_lr_gamma)#ReduceLROnPlateau(self.critic_optim, "max", threshold=10, threshold_mode="abs",
                                                     #patience=5, factor=critic_lr_gamma)
        self.eval_set = []

    def hard_update_target_networks(self):
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_log_info(self):
        return {
            "gamma" : GAMMA,
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
            "architecture": "TD3",
        }

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

    def get_action_with_noise(self, state, noise):
        action = self.actor.forward_with_noise(state, noise)
        return action  # torch.FloatTensor(action.reshape(1, -1)).to(self.device)

    def normalize_state(self, state):
        return self.normalizer.NormalizeState(state)

    def add_to_memory(self, state, action, next_state, reward, done):
        d = lambda x: 0 if x is True else 1
        self.mem.push(self.normalize_state(state), action, self.normalize_state(next_state), reward,
                      torch.FloatTensor([d(done)]))

    def update_parameters(self, iters, batch_size=None, mem=None):
        c_loss = 0
        a_loss = 0
        if batch_size is None:
            batch_size = self.batch_size
        if mem is None: mem = self.mem
        scaler = GradScaler()
        for i in range(iters):
            if len(mem) < batch_size:
                return 0, 0
            batch = Transition(*zip(*mem.sample(batch_size)))
            states = torch.stack(batch.state).to(self.device).float()
            next_states = torch.stack(batch.next_state).to(self.device).float()
            actions = torch.stack(batch.action).to(self.device).float()
            rewards = torch.stack(batch.reward).to(self.device).float()
            dones = torch.stack(batch.done).to(self.device).float()
            with autocast():
                noise = torch.Tensor(actions).data.normal_(0, 0.2).to(self.device)
                noise = noise.clamp(-0.5, 0.5)
                next_actions = self.actor_target.forward(states)
                next_actions = torch.clamp(next_actions + noise, -1, 1)

            with autocast():
                QVal, QVal2 = self.critic((states, actions))
                next_Q1, next_Q2 = self.critic_target((next_states, next_actions.detach()))
                next_Q_min = torch.min(next_Q1, next_Q2)
                QPrime = rewards + dones * GAMMA * next_Q_min
                QPrime = QPrime.detach()

                closs1 = self.criterion(QVal, QPrime).float()
                closs2 = self.criterion(QVal2, QPrime).float()

                critic_loss = closs1 + closs2

                self.critic_optim.zero_grad()
                scaler.scale(critic_loss).backward()
                scaler.unscale_(self.critic_optim)
                clip_grad_norm_(self.critic.parameters(), max_norm=10000)
                scaler.step(self.critic_optim)
                scaler.update()

                if self.update_steps % self.policy_freq == 0:
                    # Actor loss
                    actor_loss = self.critic.q1((states, self.actor.forward(states)))
                    self.actor_loss = -actor_loss.mean()
                    self.actor_optim.zero_grad()
                    scaler.scale(self.actor_loss).backward()
                    scaler.unscale_(self.actor_optim)
                    clip_grad_norm_(self.actor.parameters(), max_norm=10000)
                    scaler.step(self.actor_optim)
                    scaler.update()

                    if self.config is None:
                        self.soft_update(self.actor_target, self.actor, TAU)
                    else:
                        self.soft_update(self.actor_target, self.actor, self.config.tau)

            if self.config is None:
                self.soft_update(self.critic_target, self.critic, TAU)
            else:
                self.soft_update(self.critic_target, self.critic, self.config.tau)

            self.critic_lr_scheduler.step()
            self.actor_lr_scheduler.step()
            self.update_steps += 1
            c_loss += critic_loss.item()
            a_loss += self.actor_loss.item()
        return c_loss / iters, a_loss / iters


    def save_model(self, filename=FILE_NAME, directory=FILE_LOCATION):
        if not os.path.exists(directory + "/" + filename):
            os.mkdir(directory + "/" + filename)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}/agent_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}/agent_critic.pth")

    def save(self, timestep, plotter, filename=FILE_NAME, directory=FILE_LOCATION):
        self.save_model()
        if plotter is not None:
            statistics = plotter.save()
        else:
            statistics = {}
        with open(f"{directory}/{filename}/statistics.json", "w") as outfile:
            json.dump(statistics, outfile, cls=NumpyArrayEncoder)
        with open(f"{directory}/{filename}/memory.json", "w") as outfile:
            outfile.write(self.mem.to_json())
        with open(f"{directory}/{filename}/core.json", "w") as outfile:
            json.dump({
                "episode": timestep
            }, outfile)

    def load_model(self, filename=FILE_NAME, directory=FILE_LOCATION):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic.pth"))
        self.hard_update_target_networks()

    def load(self, timesteps, filename=FILE_NAME, directory=FILE_LOCATION):
        with open(f"{directory}/{filename}/statistics.json", "r") as infile:
            stats = json.load(infile)
            plotter = Model_Plotter(timesteps)
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
            plotter.load(np_stats)
        with open(f"{directory}/{filename}/memory.json", "r") as infile:
            self.mem.from_json(infile.read(), DEVICE)
        with open(f"{directory}/{filename}/core.json", "r") as infile:
            episode = json.load(infile)["episode"]
        self.load_model(filename, directory)
        return episode, plotter

    def create_eval_set(self, venv, eval_episodes=10):
        eval_set = []
        for i in range(eval_episodes):
            state, dist, min_dist = venv.reset()
            state = torch.FloatTensor(state, device=self.device)
            eval_set.append([state, dist, min_dist])
        self.eval_set = eval_set


    def evaluate(self, venv, rewardfunc, resetfunc, eval_episodes=10):
        avg_reward = 0
        achieve_rate = 0
        for i in range(eval_episodes):
            resetfunc()
            es = self.eval_set[i]
            state, startdist, min_dist = es[0], es[1], es[2]
            venv.reset(load_state=state.cpu().detach().numpy())
            ep_reward = 0
            for step in range(100):
                action = self.actor(state).cpu().data.numpy().flatten()
                next_state, collision, done, achieved_goal, dist_traveled, min_dist = venv.step(action)
                reward, vw, avw, tw, aw, caw, cw = rewardfunc(next_state[1], min_dist, next_state[0], action[0], action[1], step)
                ep_reward += reward
                state = torch.FloatTensor(next_state, device=self.device)
                if done:
                    break
                if achieved_goal:
                    achieve_rate += 1
            avg_reward += ep_reward
        avg_reward = avg_reward / eval_episodes
        return avg_reward, achieve_rate*100/eval_episodes


class SAC(object):
    # Define the SAC network architectures
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(SAC.Actor, self).__init__()
            self.max_action = max_action
            self.fc1 = nn.Linear(state_dim, ACTOR_LAYER_1)
            self.fc2 = nn.Linear(ACTOR_LAYER_1, ACTOR_LAYER_2)
            self.fc3 = nn.Linear(ACTOR_LAYER_2, action_dim)

        def sample(self, state):
            mean = self(state)
            log_std = torch.zeros_like(mean)
            return mean, log_std

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return self.max_action * torch.tanh(x)

        def forward_with_noise(self, state, noise):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = torch.tanh(x)
            x = noise.get_action(x)
            x = torch.FloatTensor(x, device=DEVICE)
            return self.max_action * torch.tanh(x)

        '''def sample(self, state):
            mean = self(state)
            log_std = torch.zeros_like(mean)  # You can modify this to output log_std from your actor network

            # Reparameterization trick
            normal = torch.distributions.Normal(mean, log_std.exp())
            action = normal.rsample()

            # Enforce action bounds (e.g., if you have action limits)
            action = torch.tanh(action)  # Apply tanh to squash the action

            # Compute log probability of the action
            log_prob = normal.log_prob(action).sum(1, keepdim=True) - torch.log(1 - action.pow(2) + 1e-6).sum(1,
                                                                                                              keepdim=True)

            return action, log_prob'''

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SAC.Critic, self).__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, CRITIC_LAYER_1)
            self.fc2 = nn.Linear(CRITIC_LAYER_1, CRITIC_LAYER_2)
            self.fc3 = nn.Linear(CRITIC_LAYER_2, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self, state_dim, action_dim, device, inpmap, batch_size, config=None, max_action=1, alpha=0.2):
        self.device = device
        self.map = inpmap
        self.batch_size = batch_size
        if config is None:
            self.config = None
            self.actor_lr = ACTOR_LR
            self.critic_lr = CRITIC_LR
            actor_lr_gamma = ACTOR_LR_GAMMA
            critic_lr_gamma = CRITIC_LR_GAMMA
        else:
            self.config = config
            self.actor_lr = config.actor_lr
            self.critic_lr = config.critic_lr
            actor_lr_gamma = config.actor_lr_gamma
            critic_lr_gamma = config.critic_lr_gamma


        self.actor = self.Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_actor = self.Actor(state_dim, action_dim, max_action).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic1 = self.Critic(state_dim, action_dim).to(self.device)
        self.critic2 = self.Critic(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

        self.target_critic1 = self.Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = self.Critic(state_dim, action_dim).to(self.device)
        self.hard_update_target_networks()

        self.max_action = max_action
        self.target_entropy = -action_dim
        self.alpha = alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.actor_lr)
        self.alpha = torch.tensor([alpha], requires_grad=True)
        self.mem = ReplayMemory(10000000)
        self.normalizer = Normalizer(inpmap)
        self.criterion = f.mse_loss

        self.actor_lr_scheduler = ReduceLROnPlateau(self.actor_optimizer, "max", threshold=10, threshold_mode="abs",
                                                    patience=5, factor=actor_lr_gamma)
        self.critic_lr_scheduler = ReduceLROnPlateau(self.critic1_optimizer, "max", threshold=10, threshold_mode="abs",
                                                        patience=5, factor=critic_lr_gamma)

        self.eval_set = []

    def get_log_info(self):
        return {
            "gamma" : GAMMA,
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
            "architecture": "SAC",
        }

    def hard_update_target_networks(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state, device=self.device)
        action = self.actor(state).cpu().data.numpy()
        return action

    def sample_action(self, state):
        with torch.no_grad():
            mean, log_std = self.actor.sample(state)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state):
        with torch.no_grad():
            action = self.actor.forward(state)
        return action

    def get_action_with_noise(self, state, noise):
        with torch.no_grad():
            action = self.actor.forward_with_noise(state, noise)
        return action

    def normalize_state(self, state):
        return self.normalizer.NormalizeState(state)

    def add_to_memory(self, state, action, next_state, reward, done):
        d = lambda x: 0 if x is True else 1
        self.mem.push(self.normalize_state(state), action, self.normalize_state(next_state), reward,
                      torch.FloatTensor([d(done)]))

    def update_parameters(self, iters, batch_size=None, mem=None):
        if mem is None: mem = self.mem
        if batch_size is None:
            batch_size = self.batch_size
        if len(mem) < batch_size:
            return 0, 0
        batch = Transition(*zip(*mem.sample(batch_size)))
        states = torch.stack(batch.state).to(self.device).float()
        next_states = torch.stack(batch.next_state).to(self.device).float()
        actions = torch.stack(batch.action).to(self.device).float()
        rewards = torch.stack(batch.reward).to(self.device).float()
        dones = torch.stack(batch.done).to(self.device).float()
        # Compute the target Q values
        with torch.no_grad():
            next_action, next_log_pi = self.sample_action(next_states)
            target_Q1 = self.target_critic1(next_states, next_action)
            target_Q2 = self.target_critic2(next_states, next_action)
            alpha = self.alpha.item()
            target_V = torch.min(target_Q1, target_Q2) - alpha * next_log_pi
            target_Q = rewards + (1 - dones) * GAMMA * target_V

        # Compute the current Q values
        current_Q1 = self.critic1.forward(states, actions)
        current_Q2 = self.critic2.forward(states, actions)

        # Compute the actor loss
        new_action, log_pi = self.sample_action(states)
        actor_loss = (alpha * log_pi - torch.min(self.critic1(states, new_action),
                                                 self.critic2(states, new_action))).mean()

        # Compute the critic loss
        critic1_loss = self.criterion(current_Q1, target_Q)
        critic2_loss = self.criterion(current_Q2, target_Q)

        # Update the actor and critic networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update the temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_pi.cpu() + self.target_entropy).detach()).mean().to(self.device)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Update the target networks
        if self.config is None:
            self.soft_update(self.target_actor, self.actor, TAU)
            self.soft_update(self.target_critic1, self.critic1, TAU)
            self.soft_update(self.target_critic2, self.critic2, TAU)
        else:
            self.soft_update(self.target_actor, self.actor, self.config.tau)
            self.soft_update(self.target_critic1, self.critic1, self.config.tau)
            self.soft_update(self.target_critic2, self.critic2, self.config.tau)

        return actor_loss.item(), critic1_loss.item()

    def create_eval_set(self, venv, eval_episodes=10):
        eval_set = []
        for i in range(eval_episodes):
            state, dist, min_dist = venv.reset()
            state = torch.FloatTensor(state, device=self.device)
            eval_set.append([state, dist, min_dist])
        self.eval_set = eval_set


    def evaluate(self, venv, rewardfunc, resetfunc, eval_episodes=10):
        avg_reward = 0
        achieve_rate = 0
        for i in range(eval_episodes):
            resetfunc()
            es = self.eval_set[i]
            state, startdist, min_dist = es[0], es[1], es[2]
            venv.reset(load_state=state.cpu().detach().numpy())
            ep_reward = 0
            for step in range(100):
                action = self.actor(state).cpu().data.numpy().flatten()
                next_state, collision, done, achieved_goal, dist_traveled, min_dist = venv.step(action)
                reward, vw, avw, tw, aw, caw, cw = rewardfunc(next_state[1], min_dist, next_state[0], action[0], action[1], step)
                ep_reward += reward
                state = torch.FloatTensor(next_state, device=self.device)
                if done:
                    break
                if achieved_goal:
                    achieve_rate += 1
            avg_reward += ep_reward
        avg_reward = avg_reward / eval_episodes
        return avg_reward, achieve_rate*100/eval_episodes

    def save_model(self, filename=FILE_NAME, directory=FILE_LOCATION):
        if not os.path.exists(directory + "/" + filename):
            os.mkdir(directory + "/" + filename)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}/agent_actor.pth")
        torch.save(self.critic1.state_dict(), f"{directory}/{filename}/agent_critic1.pth")
        torch.save(self.critic2.state_dict(), f"{directory}/{filename}/agent_critic2.pth")

    def load_model(self, filename=FILE_NAME, directory=FILE_LOCATION):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"{directory}/{filename}/agent_critic2.pth"))
        self.hard_update_target_networks()

    def save(self, timestep, plotter, filename=FILE_NAME, directory=FILE_LOCATION):
        self.save_model()
        if plotter is not None:
            statistics = plotter.save()
        else:
            statistics = {}
        with open(f"{directory}/{filename}/statistics.json", "w") as outfile:
            json.dump(statistics, outfile, cls=NumpyArrayEncoder)
        with open(f"{directory}/{filename}/memory.json", "w") as outfile:
            outfile.write(self.mem.to_json())
        with open(f"{directory}/{filename}/core.json", "w") as outfile:
            json.dump({
                "episode": timestep
            }, outfile)

    def load(self, timesteps, filename=FILE_NAME, directory=FILE_LOCATION):
        with open(f"{directory}/{filename}/statistics.json", "r") as infile:
            stats = json.load(infile)
            plotter = Model_Plotter(timesteps)
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
            plotter.load(np_stats)
        with open(f"{directory}/{filename}/memory.json", "r") as infile:
            self.mem.from_json(infile.read(), DEVICE)
        with open(f"{directory}/{filename}/core.json", "r") as infile:
            episode = json.load(infile)["episode"]
        self.load_model(filename, directory)
        return episode, plotter
