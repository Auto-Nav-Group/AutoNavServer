import lightning
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
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

    def q1(self, state,action):
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
class TD3(pl.LightningModule):
    def __init__(self, state_dim, action_dim, env, reward_function):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.env = env
        self.reward_function = reward_function
        self.mem = ReplayMemory(1000000)
        self.automatic_optimization = False
        self.bool_conv = lambda x: 1 if x else 0
        self.achieve_goal_conv = lambda x: 1 if x else 0
    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        return [actor_optimizer, critic_optimizer]

    def backward(self, loss, **kwargs):
        loss[0].backward()
        loss[1].backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def training_step(self, batch, batch_idx):
        state, _, _ = self.env.reset()
        self.reward_function.reset()
        total_reward = 0
        achieved_goal = False
        for _ in range(100):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            action = self.actor(state)
            next_state, collision, done, achieve_goal, _, _ = self.env.step(action)
            reward = self.reward_function.get_reward(next_state[1])
            total_reward+=reward
            next_state = torch.tensor(next_state, device=self.device)
            self.mem.push(state, action, next_state, torch.tensor(reward, device=self.device), torch.tensor(self.bool_conv(done), device=self.device))
            state = next_state
            achieved_goal=achieve_goal
            if done: break
        batch = self.mem.sample(100)
        self.log('reward', total_reward)
        self.log('achieve', self.achieve_goal_conv(achieved_goal))
        state_batch = torch.stack([x.state for x in batch])
        action_batch = torch.stack([x.action for x in batch])
        next_state_batch = torch.stack([x.next_state for x in batch])
        reward_batch = torch.stack([x.reward for x in batch])
        done_batch = torch.stack([x.done for x in batch])
        q1, q2 = self.critic(state_batch, action_batch)
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            q_next = torch.min(q1_next, q2_next)
            q_target = reward_batch + 0.99*q_next*(1-done_batch)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.log('critic_loss', critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss = critic_loss.to(dtype=torch.float16)
        actor_loss = -self.critic(state_batch, self.actor(state_batch))[0].mean()
        self.log('actor_loss', actor_loss)
        self.actor_optimizer.zero_grad()
        return torch.Tensor([critic_loss, actor_loss])

    def on_epoch_end(self):
        # Update target networks at the end of each epoch
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

if __name__ == "__main__":
    state_dim = 10
    action_dim = 2
    path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\BasicMap.json'
    JSON = json.load(open(path))
    mapobj = Map(JSON)
    env = RobotVEnv(map=mapobj, assets_path=ASSET_PATH)
    data = np.zeros(1)
    wandblogger = WandbLogger(project='anv-2', entity='auto-nav-group', log_model=True)
    trainer = lightning.Trainer(max_epochs=10000, accelerator='gpu', precision=16, logger=wandblogger, log_every_n_steps=1)
    rew = SimpleReward()
    model = TD3(state_dim, action_dim, env, rew)
    trainer.fit(model = model, train_dataloaders = data)