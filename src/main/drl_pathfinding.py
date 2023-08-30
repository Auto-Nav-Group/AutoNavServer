import numpy as np



COUNT_THRESHOLD = 100 # Threshold for the amount of steps
COL_REWARD = -90 # Minimum reward

def evaluate(network, epoch, eval_episodes=10):   # Evaluate episode measures the performance of the agent over the amount of episodes.
    avg_reward = 0 # Average reward
    col = 0 # Amount of collissions

    for ep in range(eval_episodes): # Loop through the amount of episodes
        count = 0 # Count the amount of steps
        state = None# TODO: Reset the environment
        done = False # Termination flag

        while not done and count < COUNT_THRESHOLD:
            action = network.get_action(np.array(state)) # Get the action from the agent
            next_state, reward, done, ep = None # TODO: Perform the action in the environment
            state = next_state # Update the state
            avg_reward += reward # Update the average reward
            count += 1 # Update the step count
            if reward < COL_REWARD:
                col += 1
    avg_reward /= eval_episodes # Calculate the average reward over the evaluation episodes
    avg_col = col / eval_episodes # Calculate the average amount of collissions over the evaluation episodes

    print ("---------------------------------------",
           "Epoch: %d" % epoch,
              "Average Reward: %f" % avg_reward,
                "Average Collisions: %f" % avg_col,
                    "---------------------------------------")
    return avg_reward