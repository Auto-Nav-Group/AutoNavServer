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
from drl_utils import ReplayMemory, Transition, NumpyArrayEncoder, Model_Plotter
from drl_venv import MAX_SPEED, TIME_DELTA
from IPython import display
import torch.optim as optim
import random

FILE_LOCATION = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\models"
FILE_NAME = "SampleModel"

ONE_SITUATION = True

STATE_DIM = 8
ACTION_DIM = 2

SIMPLE_LAYER_1 = 512
SIMPLE_LAYER_2 = 512
SIMPLE_LAYER_3 = 512
SIMPLE_LAYER_4 = 256
SIMPLE_LAYER_5 = 256
SIMPLE_LAYER_6 = 256

COLLISION_WEIGHT = -150
TIME_WEIGHT = 0#-6
FINISH_WEIGHT = 150
DIST_WEIGHT = 0#2
PASS_DIST_WEIGHT = 0
CHALLENGE_WEIGHT = 10
CHALLENGE_EXP_BASE = 1.25
ANGLE_WEIGHT = 0#-2

EPISODES = 40000
TIMESTEP_CAP = 100
SAVE_FREQ = 100 #Save model every x episodes
IDEAL_PROBABILITY = 0.9
IDEAL_ANGLE_SWITCH = np.pi/64
START_ANGLE = np.pi#np.pi/32

BATCH_SIZE = 256 #Batch size for training
GAMMA = 0.999 #Discount factor
EPS_START = 0.9
EPS_END = 0.05 #Final value of epsilon
EPS_DECAY = 1000 #Rate of exponential decay of epsilon. Higher is lower.
Tau = 0.0005 #Update rate of target network
LR = 1e-5 #Learning rate of optimizer

MEMORY = ReplayMemory(100000)

TRAINING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward(done, collision, timestep, achieved_goal, delta_dist, initdistance, initangle, ovr_dist, angle):
    dist_weight = delta_dist*DIST_WEIGHT
    angle_weight = (abs(angle)/np.pi) * ANGLE_WEIGHT
    #time_weight = TIME_WEIGHT*(initdistance/MAX_SPEED-timestep/TIME_DELTA)
    time_weight = 0
    total_weight = dist_weight + angle_weight + time_weight
    if (ovr_dist > initdistance):
        total_weight += PASS_DIST_WEIGHT
    if initdistance/MAX_SPEED < timestep*TIME_DELTA:
        total_weight += TIME_WEIGHT
        time_weight = TIME_WEIGHT
    if done:
        if achieved_goal:
            initangle = abs(initangle)/np.pi
            dist_challenge = CHALLENGE_EXP_BASE**initdistance
            angle_challenge = CHALLENGE_EXP_BASE**initangle
            return FINISH_WEIGHT + time_weight + dist_challenge*CHALLENGE_WEIGHT + angle_challenge*CHALLENGE_WEIGHT, time_weight, dist_weight, angle_weight
        elif collision is not True:
            return COLLISION_WEIGHT + time_weight, time_weight, dist_weight, angle_weight
    if collision is True:
        return COLLISION_WEIGHT + time_weight, time_weight, dist_weight, angle_weight
    return total_weight, time_weight, dist_weight, angle_weight

class SimpleActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimpleActor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, SIMPLE_LAYER_1)
        self.layer_2 = nn.Linear(SIMPLE_LAYER_1, SIMPLE_LAYER_2)
        self.layer_3 = nn.Linear(SIMPLE_LAYER_2, SIMPLE_LAYER_3)
        self.layer_4 = nn.Linear(SIMPLE_LAYER_3, SIMPLE_LAYER_4)
        self.layer_5 = nn.Linear(SIMPLE_LAYER_4, SIMPLE_LAYER_5)
        self.layer_6 = nn.Linear(SIMPLE_LAYER_5, SIMPLE_LAYER_6)
        self.layer_7 = nn.Linear(SIMPLE_LAYER_6, action_dim)

    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = F.relu(self.layer_3(a))
        a = F.relu(self.layer_4(a))
        a = F.relu(self.layer_5(a))
        a = F.relu(self.layer_6(a))
        a = self.layer_7(a)
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

def train(env, agent=None, plotter=None, prev_episode=0):
    steps_done = 0
    def get_action(c_state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q = agent(c_state)
                qmax = q.max(0)[0]
                return qmax.view(1)
        else:
            return torch.tensor([random.uniform(-np.pi, np.pi)], device=TRAINING_DEVICE,
                                dtype=torch.float32)

    '''def plot_results():
        line.set_xdata(x)
        line.set_ydata(y)  # Update the y-data only
        dist.set_xdata(x)  # Update the x-data for dist
        dist.set_ydata(distances)  # Update the y-data for dist
        rewards_plot.set_xdata(total_rewards_x)  # Update the x-data for rewards_plot
        rewards_plot.set_ydata(total_rewards)  # Update the y-data for rewards_plot
        ax[0].set_xlim(0, episode)
        ax[1].set_xlim(0, episode)
        ax[2].set_xlim(0, 100)
        ax[0].set_ylim(np.min(y) - 10, np.max(y) + 10)
        ax[1].set_ylim(np.min(distances) - 10, np.max(distances) + 10)
        ax[2].set_ylim(np.min(total_rewards) - 10, np.max(total_rewards) + 10)
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.pause(0.1)'''

    def optimize():
        if len(MEMORY) < BATCH_SIZE:
            return
        transitions = MEMORY.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=TRAINING_DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        action_batch = action_batch.type(torch.LongTensor).to(TRAINING_DEVICE)

        action_batch = action_batch.clamp(0, SIMPLE_LAYER_2 - 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        action_vals = agent(state_batch)
        state_action_values = action_vals.gather(0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=TRAINING_DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = targetagent(non_final_next_states).max(1)[0].to(TRAINING_DEVICE)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.to(TRAINING_DEVICE) * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
        optimizer.step()

    if agent is None:
        agent = SimpleActor(STATE_DIM, ACTION_DIM).to(TRAINING_DEVICE)
    prev = 0
    targetagent = SimpleActor(STATE_DIM, ACTION_DIM).to(TRAINING_DEVICE)
    targetagent.load_state_dict(agent.state_dict())

    agent.to(TRAINING_DEVICE)

    if plotter is None:
        plotter = Model_Plotter(EPISODES)

    '''total_rewards = np.zeros(100)
    total_rewards_x = np.arange(100)
    distances = np.empty(EPISODES)'''

    optimizer = optim.AdamW(agent.parameters(), lr=LR, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    '''x = np.arange(EPISODES)  # Use np.arange to create a proper x-axis
    y = np.zeros(EPISODES)   # Initialize y with zeros'''

    episode = prev_episode

    '''if statistics is not None and len(statistics) == 5:
        x = statistics[0]
        y = statistics[1]
        distances = statistics[2]
        total_rewards = statistics[3]
        total_rewards_x = statistics[4]
        if episode>99:
            prev = total_rewards[99]

    figure, ax = plt.subplots(3, figsize=(10, 6))

    line, = ax[0].plot(x, y, 'o', color='tab:blue')  # Use x and y here
    dist, = ax[1].plot(x, y, 'o', color='tab:red')   # Use x and y here
    rewards_plot, = ax[2].plot(x, y, 'o', color='tab:green')  # Use x and y here

    plt.title("Training Results")

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.show()

    plot_results()'''


    ideal_angle = START_ANGLE

    initial_state = env.reset(ideal_angle)

    ideal_angle_episode = episode

    for episode in range(episode, EPISODES-1):
        current_percent = plotter.get_ideal_probability(ideal_angle_episode, episode)
        if current_percent >= IDEAL_PROBABILITY and ideal_angle <= np.pi:
            ideal_angle += IDEAL_ANGLE_SWITCH
            ideal_angle_episode = 0
        else:
            ideal_angle_episode = episode
        if ONE_SITUATION:
            state = env.reload(initial_state, ideal_angle)
        else:
            state = env.reset(ideal_angle)
        initdist = state[0]
        initangle = state[1]
        done = False
        achieved_goal = False
        timestep = 0
        episode_reward = 0
        episode_tw = 0
        episode_dw = 0
        episode_aw = 0
        episode_achieve = 0
        episode_collide = 0
        log_probs = []
        rewards = []
        ovr_dist = 0

        while (done or timestep > TIMESTEP_CAP) is False:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(TRAINING_DEVICE)
            action = get_action(state_tensor)
            next_state, collision, done, achieved_goal, dist_traveled = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(TRAINING_DEVICE)
            ovr_dist += dist_traveled
            if collision is True:
                done = True
                episode_collide = 1
            if achieved_goal is True:
                episode_achieve = 1
            if done == True:
                next_state = None
            reward = 0
            tw, dw, aw = 0, 0, 0
            if (next_state is not None):
                reward, tw, dw, aw  = get_reward(done, collision, timestep, achieved_goal, (initdist-next_state[0])/initdist, initdist, initangle, ovr_dist, next_state[1])
            else:
                reward, tw, dw, aw = get_reward(done, collision, timestep, achieved_goal, initangle, initdist, initangle, 0, 0)
            episode_reward += reward
            episode_tw += tw
            episode_dw += dw
            episode_aw += aw
            rewards.append(reward)
            MEMORY.push(state_tensor, action, next_state, torch.tensor([reward]).to(TRAINING_DEVICE))
            state = next_state
            timestep += 1

            optimize()

            targetagent_dict = targetagent.state_dict()
            agent_dict = agent.state_dict()
            for key in agent_dict:
                targetagent_dict[key] = Tau * agent_dict[key] + (1 - Tau) * targetagent_dict[key]


        clip_grad_norm_(agent.parameters(), 0.5)


        '''avg = sum(total_rewards)/(episode+1)

        if episode>100:
            recent_rewards = total_rewards
            for i in range(len(total_rewards)):
                if i<episode-100:
                    recent_rewards = np.delete(recent_rewards, 0)
                    recent_rewards = np.resize(recent_rewards, 100)
                    break
            avg = sum(recent_rewards)/100'''


        print(f"Episode {episode + 1}, Episode Reward: {episode_reward}")

        if SAVE_FREQ != -1 and (episode+1) % SAVE_FREQ == 0:
            save(agent, FILE_NAME, FILE_LOCATION, plotter, episode, MEMORY)

        plotter.update(episode, initdist, episode_reward, episode_dw, episode_aw, episode_tw, episode_achieve, episode_collide)


        '''y.put(episode, avg)
        distances.put(episode, initdist)

        if total_rewards[99] == prev:
            total_rewards = np.delete(total_rewards, 0)
            total_rewards = np.resize(total_rewards, 100)
            if type(episode_reward) is torch.Tensor:
                prev = episode_reward.cpu().detach().numpy()
                total_rewards.put(99, episode_reward.cpu().detach().numpy())
            else:
                prev = episode_reward
                total_rewards.put(99, episode_reward)
        else:
            if type(episode_reward) is torch.Tensor:
                prev = episode_reward.cpu().detach().numpy()
                total_rewards.put(episode, episode_reward.cpu().detach().numpy())
            else:
                prev = episode_reward
                total_rewards.put(episode, episode_reward)
        plot_results()'''




def train_load(env):
    agent = SimpleActor(STATE_DIM, ACTION_DIM)
    agent, stats, episode = load(agent, FILE_NAME, FILE_LOCATION)
    train(env, agent, stats, episode)

def save(agent, filename, directory, plotter, episode, memory):
    if not os.path.exists(directory + "\\" + filename):
        os.mkdir(directory + "\\" + filename)
    statistics = plotter.save()
    with open(f"{directory}/{filename}/statistics.json", "w") as outfile:
        json.dump(statistics, outfile, cls=NumpyArrayEncoder)
    with open(f"{directory}/{filename}/memory.json", "w") as outfile:
        outfile.write(memory.to_json())
    with open(f"{directory}/{filename}/core.json", "w") as outfile:
        json.dump({
            "episode" : episode
        }, outfile)
    torch.save(agent.state_dict(), f"{directory}/{filename}/agent_actor.pth")

def load(agent, filename, directory):
    with open(f"{directory}/{filename}/statistics.json", "r") as infile:
        stats = json.load(infile)
        plotter = Model_Plotter(EPISODES)
        np_stats = {
            "avg_y" : np.asarray(stats["avg_y"]),
            "dist_y" : np.asarray(stats["dist_y"]),
            "total_reward_y" : np.asarray(stats["total_reward_y"]),
            "dist_weight_y" : np.asarray(stats["dist_weight_y"]),
            "angle_weight_y" : np.asarray(stats["angle_weight_y"]),
            "time_weight_y" : np.asarray(stats["time_weight_y"]),
            "achieve_history" : np.asarray(stats["achieve_history"]),
            "collision_history" : np.asarray(stats["collision_history"]),
            "none_history" : np.asarray(stats["none_history"]),
            "achieve_chance_y" : np.asarray(stats["achieve_chance_y"]),
            "collision_chance_y" : np.asarray(stats["collision_chance_y"]),
            "none_chance_y" : np.asarray(stats["none_chance_y"])
        }
        plotter.load(stats)
    with open(f"{directory}/{filename}/memory.json", "r") as infile:
        MEMORY.from_json(infile.read(), TRAINING_DEVICE)
    with open(f"{directory}/{filename}/core.json", "r") as infile:
        episode = json.load(infile)["episode"]
    agent.load_state_dict(torch.load(f"{directory}/{filename}/agent_actor.pth"))
    agent.to(TRAINING_DEVICE)
    return agent, plotter, episode