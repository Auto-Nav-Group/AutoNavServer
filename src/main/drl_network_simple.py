import torch
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

FILE_LOCATION = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\models"
FILE_NAME = "SampleModel"

SIMPLE_LAYER_1 = 10
SIMPLE_LAYER_2 = 5

COLLISION_WEIGHT = -1
TIME_WEIGHT = -0.1
FINISH_WEIGHT = 100

EPISODES = 125
TIMESTEP_CAP = 10000

NOISE = random.uniform(0, 0.5)

TRAINING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward(collisions, timesteps, done, achieved_goal):
    if done:
        if achieved_goal:
            return FINISH_WEIGHT
    return collisions*COLLISION_WEIGHT + timesteps*TIME_WEIGHT

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
        R = r +0.99*R
        returns.insert(0,R)
    returns = torch.tensor(returns).to(TRAINING_DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train(env, agent=None):
    if agent is None:
        agent = SimpleActor(4, 2)

    totalrewards=[]

    optimizer = torch.optim.Adam(agent.parameters())
    plt.ion()

    x = np.array(range(EPISODES))
    y = np.array(range(EPISODES))

    figure, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, EPISODES)

    line, = ax.plot(0, 0, 'o', color='tab:blue')

    plt.title("Training Results")

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.show()

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        achieved_goal = False
        timestep = 0
        episode_reward = 0
        log_probs = []
        rewards = []

        while (done or timestep > TIMESTEP_CAP) is False:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(TRAINING_DEVICE)
            action = agent(state_tensor)
            noise = torch.Tensor(action).data.normal_(0, 0.5).to(TRAINING_DEVICE)
            action = action + noise
            noise = random.uniform(0, 0.5)
            next_state, collision, done, achieved_goal = env.step(action)
            reward = get_reward(collision, done, timestep, achieved_goal)
            episode_reward += reward
            log_prob = -0.5 * (action - agent(state_tensor)).pow(2).sum()  # Simplified log prob
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            timestep += 1

        if not achieved_goal:
            episode_reward -= FINISH_WEIGHT

        totalrewards.append(episode_reward)

        returns = compute_returns(rewards)
        loss = torch.stack(log_probs) * returns
        loss = -loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Episode Reward: {episode_reward}, Average Reward: {np.mean(rewards)}")
        save(agent, FILE_NAME, FILE_LOCATION)

        x.put(episode, episode)
        y.put(episode, episode_reward)

        line.set_xdata(x)
        line.set_ydata(y)
        ax.set_ylim(np.min(y) - 10, np.max(y) + 10)
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.show()
        time.sleep(0.1)

def train_load(env):
    agent = SimpleActor(4, 2)
    agent = load(agent, FILE_NAME, FILE_LOCATION)
    train(env, agent)
def save(agent, filename, directory):
    torch.save(agent.state_dict(), "%s/%s_actor.pth" % (directory, filename))

def load(agent, filename, directory):
    agent.load_state_dict(
        torch.load("%s/%s_actor.pth" % (directory, filename))
    )
    return agent