import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import random

FILE_LOCATION = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\models"
FILE_NAME = "SampleModel"

SIMPLE_LAYER_1 = 25
SIMPLE_LAYER_2 = 25

COLLISION_WEIGHT = -1
TIME_WEIGHT = -0.01
FINISH_WEIGHT = 1000

EPISODES = 40000
TIMESTEP_CAP = 10000

GAMMA = 0.99
NOISE = random.uniform(0, 0.1)

TRAINING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward(done, collision, timestep, achieved_goal):
    if done:
        if achieved_goal:
            return FINISH_WEIGHT + TIME_WEIGHT * timestep
    if collision is True:
        return COLLISION_WEIGHT - TIME_WEIGHT * timestep
    return 0

class SimpleActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimpleActor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, SIMPLE_LAYER_1)
        self.layer_2 = nn.Linear(SIMPLE_LAYER_1, SIMPLE_LAYER_2)
        self.layer_3 = nn.Linear(SIMPLE_LAYER_2, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = self.layer_3(a)
        a = self.tanh(a)
        return a

def compute_returns(rewards):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(TRAINING_DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train(env, agent=None):
    if agent is None:
        agent = SimpleActor(4, 2).to(TRAINING_DEVICE)

    agent.to(TRAINING_DEVICE)

    total_rewards = np.zeros(EPISODES)
    distances = np.empty(EPISODES)

    optimizer = optim.AdamW(agent.parameters())
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

    plt.ion()

    x = np.arange(EPISODES)  # Use np.arange to create a proper x-axis
    y = np.zeros(EPISODES)   # Initialize y with zeros

    figure, ax = plt.subplots(3, figsize=(10, 6))

    line, = ax[0].plot(x, y, 'o', color='tab:blue')  # Use x and y here
    dist, = ax[1].plot(x, y, 'o', color='tab:red')   # Use x and y here
    rewards_plot, = ax[2].plot(x, y, 'o', color='tab:green')  # Use x and y here

    plt.title("Training Results")

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.show()

    initial_state = env.reset()

    for episode in range(EPISODES):
        state = initial_state
        initdist = state[0]
        done = False
        achieved_goal = False
        timestep = 0
        episode_reward = 0
        log_probs = []
        rewards = []

        while (done or timestep > TIMESTEP_CAP) is False:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(TRAINING_DEVICE)
            action = agent(state_tensor)
            noise = torch.normal(0, 0.1, size=action.shape).to(TRAINING_DEVICE)
            #action = torch.clamp(action + noise, -1, 1)  # Add noise and clamp to [-1, 1]
            next_state, collision, done, achieved_goal = env.step(action)
            if collision is True:
                done = True
            reward = get_reward(done, collision, timestep, achieved_goal)
            episode_reward += reward
            log_prob = -0.5 * (action - agent(state_tensor)).pow(2).sum()
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            timestep += 1

        total_rewards[episode] = episode_reward

        returns = compute_returns(rewards)
        loss = torch.stack(log_probs) * returns
        loss = -loss.mean()  # Use mean instead of sum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        clip_grad_norm_(agent.parameters(), 0.5)

        avg = sum(total_rewards)/(episode+1)

        print(f"Episode {episode + 1}, Episode Reward: {episode_reward}, Average Reward: {avg}")
        save(agent, FILE_NAME, FILE_LOCATION)

        y.put(episode, avg)
        distances.put(episode, initdist)
        total_rewards.put(episode, episode_reward)

        line.set_xdata(x)
        line.set_ydata(y)  # Update the y-data only
        dist.set_xdata(x)  # Update the x-data for dist
        dist.set_ydata(distances)  # Update the y-data for dist
        rewards_plot.set_xdata(x)  # Update the x-data for rewards_plot
        rewards_plot.set_ydata(total_rewards)  # Update the y-data for rewards_plot
        ax[0].set_xlim(0, episode)
        ax[1].set_xlim(0, episode)
        ax[2].set_xlim(0, episode)
        ax[0].set_ylim(np.min(y) - 10, np.max(y) + 10)
        ax[1].set_ylim(np.min(distances) - 10, np.max(distances) + 10)
        ax[2].set_ylim(np.min(total_rewards) - 10, np.max(total_rewards) + 10)
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.pause(0.1)

def train_load(env):
    agent = SimpleActor(4, 2)
    agent = load(agent, FILE_NAME, FILE_LOCATION)
    train(env, agent)

def save(agent, filename, directory):
    torch.save(agent.state_dict(), f"{directory}/{filename}_actor.pth")

def load(agent, filename, directory):
    agent.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
    agent.to(TRAINING_DEVICE)
    return agent