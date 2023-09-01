import numpy as np
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from drl_utils import ReplayBuffer



COUNT_THRESHOLD = 100  # Threshold for the amount of steps
COL_REWARD = -90  # Minimum reward


def evaluate(network, epoch,
             eval_episodes=10):  # Evaluate episode measures the performance of the agent over the amount of episodes.
    avg_reward = 0  # Average reward
    col = 0  # Amount of collisions

    for ep in range(eval_episodes):  # Loop through the amount of episodes
        count = 0  # Count the amount of steps
        state = None  # TODO: Reset the environment
        done = False  # Termination flag

        while not done and count < COUNT_THRESHOLD:
            action = network.get_action(np.array(state))  # Get the action from the agent
            next_state, reward, done, ep = None  # TODO: Perform the action in the environment
            state = next_state  # Update the state
            avg_reward += reward  # Update the average reward
            count += 1  # Update the step count
            if reward < COL_REWARD:
                col += 1
    avg_reward /= eval_episodes  # Calculate the average reward over the evaluation episodes
    avg_col = col / eval_episodes  # Calculate the average amount of collissions over the evaluation episodes

    print("---------------------------------------",
          "Epoch: %d" % epoch,
          "Average Reward: %f" % avg_reward,
          "Average Collisions: %f" % avg_col,
          "---------------------------------------")
    return avg_reward


class Actor(nn.Module):  # Neural network
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()  # default nn init

        self.layer_1 = nn.Linear(state_dim, 800)  # Layer 1 with statedim input and 800 output neurons
        self.layer_2 = nn.Linear(800, 600)  # Layer 2 with 800 input neurons and 600 output neurons
        self.layer_3 = nn.Linear(600, action_dim)  # Layer 3 with 600 input neurons and actiondim output neurons
        self.tanh = nn.Tanh()  # Scales action values to range [-1, 1]

    def forward(self, s):  # Forward pass of neural network
        s = F.relu(self.layer_1(s))  # Rectified linear unit activation function for layer 1
        s = F.relu(self.layer_2(s))  # Rectified linear unit activation function for layer 2
        a = self.tanh(self.layer_3(s))  # Applies scale action values on layer 3 to range [-1,1]
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # First critic network
        self.layer_1 = nn.Linear(state_dim, 800)  # Linear layer for state input
        self.layer_2_s = nn.Linear(800, 600)  # Linear layer for state pathway
        self.layer_2_a = nn.Linear(action_dim, 600)  # Linear layer for action pathway
        self.layer_3 = nn.Linear(600, 1)  # Linear layer for Q-value output

        # Second critic network (for double Q-learning)
        self.layer_4 = nn.Linear(state_dim, 800)  # Linear layer for state input
        self.layer_5_s = nn.Linear(800, 600)  # Linear layer for state pathway
        self.layer_5_a = nn.Linear(action_dim, 600)  # Linear layer for action pathway
        self.layer_6 = nn.Linear(600, 1)  # Linear layer for Q-value output

    def forward(self, s, a):
        # Compute Q-values for the first critic network
        s1 = F.relu(self.layer_1(s))  # Apply ReLU to state input
        s1 = F.relu(self.layer_2_s(s1))  # Apply ReLU to state pathway
        a = F.relu(self.layer_2_a(a))  # Apply ReLU to action pathway
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())  # Matrix multiplication
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())  # Matrix multiplication
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)  # Combine and apply ReLU
        q1 = self.layer_3(s1)  # Compute Q-value for first critic

        # Compute Q-values for the second critic network (double Q-learning)
        s2 = F.relu(self.layer_4(s))  # Apply ReLU to state input
        s2 = F.relu(self.layer_5_s(s2))  # Apply ReLU to state pathway
        a = F.relu(self.layer_5_a(a))  # Apply ReLU to action pathway
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())  # Matrix multiplication
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())  # Matrix multiplication
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)  # Combine and apply ReLU
        q2 = self.layer_6(s2)  # Compute Q-value for second critic

        return q1, q2


class TD3_NET(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
        self.actor = Actor(state_dim, action_dim).to(self.device)  # Create actor network
        self.actor_targ = Actor(state_dim, action_dim).to(self.device)  # Create target actor network
        self.actor_targ.load_state_dict(self.actor.state_dict())  # Copy actor network to target actor network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())  # Adam optimizer for actor network

        self.critic = Critic(state_dim, action_dim).to(self.device)  # Create critic network
        self.critic_targ = Critic(state_dim, action_dim).to(self.device)  # Create target critic network
        self.critic_targ.load_state_dict(self.critic.state_dict())  # Copy critic network to target critic network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())  # Adam optimizer for critic network

        self.max_action = max_action  # Maximum action value
        self.sum_writer = SummaryWriter()
        self.iteration_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buf, iterations, batch_size=100, discount=1, tau=0.005, policy_noise=0.2, noise_clip=0.5,
              policy_freq=2):
        av_Q = 0
        max_Q = -np.inf
        av_loss = 0
        for iter in range(iterations):
            # Sample replay buffer
            b_state, b_next_state, b_actions, b_reward, b_done = replay_buf.sample(batch_size)
            state = torch.Tensor(b_state).to(self.device)
            action = torch.Tensor(b_actions).to(self.device)
            next_state = torch.Tensor(b_next_state).to(self.device)
            done = torch.Tensor(1 - b_done).to(self.device)
            reward = torch.Tensor(b_reward).to(self.device)

            # Obtain the estimated action from the next state with actor_target
            next_action = self.actor_targ(next_state)

            # Select action according to policy and add clipped noise
            noise = torch.Tensor(b_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_targ(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_targ(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if iter % policy_freq == 0:
                # Compute actor loss
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the critic target network parameters.
                for param, target_param in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            av_loss += critic_loss

        self.iteration_count += 1
        self.sum_writer.add_scalar('Avg Q', av_Q / iterations, self.iteration_count)
        self.sum_writer.add_scalar('Loss', av_loss / iterations, self.iteration_count)
        self.sum_writer.add_scalar('Max Q', max_Q, self.iteration_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )