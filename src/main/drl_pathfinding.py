import torch
import numpy as np
from drl_networks import Actor, Critic, evaluate, TD3_NET
#from drl_venv import DRL_VENV
from drl_utils import ReplayMemory


OUTPUT_DIR = "G:\\Projects\\AutoNav\\AutoNavServer\\output\\drl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 4
ACTION_DIM = 2
MAX_ACTION = 1
BUF_SIZE = 1000000

MAX_TIMESTEP = 5000000

class training_config():
    def __init__(self, load_model_path=None, save_model_path=None, batch_size=40, discount=0.999999, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, exploration_noise=1, exploration_min=0.1, exploration_decay_steps=500000, max_episode_steps=1000):
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self.exploration_min = exploration_min
        self.exploration_decay_steps = exploration_decay_steps
        self.max_episode_steps = max_episode_steps
def start_training_process(training_config, VENV):
    TRAINER = TD3_NET(DEVICE, STATE_DIM, ACTION_DIM, MAX_ACTION)

    REPLAY_BUF = ReplayBuffer(BUF_SIZE)
    if training_config.load_model_path is not None:
        TRAINER.load("drl_agent", training_config.load_model_path)
    evals = []
    timestep = 0
    timestep_since_eval = 0
    episode_num = 0
    episode_reward = 0
    epoch = 0



    noise = training_config.exploration_noise

    state = None

    eval_freq = 5000

    episode_timesteps = 0
    done = True
    while timestep < MAX_TIMESTEP:
        if done:
            if timestep != 0:
                TRAINER.train(REPLAY_BUF, episode_timesteps, training_config.batch_size, training_config.discount, training_config.tau, training_config.policy_noise, training_config.noise_clip, training_config.policy_freq)
            if timestep_since_eval >= eval_freq:
                timestep_since_eval %= eval_freq
                evals.append(evaluate(TRAINER, VENV, DEVICE))
                if training_config.save_model_path is not None:
                    TRAINER.save("drl_agent", training_config.save_model_path)
                np.save(OUTPUT_DIR + "\\evals", evals)
                epoch += 1
            state = VENV.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        if noise > training_config.exploration_min:
            noise = noise - ((1-training_config.exploration_min)/training_config.exploration_decay_steps)

        action = TRAINER.get_action(np.array(state))
        action = (action + np.random.normal(0, noise, size=ACTION_DIM)).clip(-MAX_ACTION, MAX_ACTION)

        #TODO: Added random exploration when near obstacles

        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, target = VENV.step(a_in)
        done_bool = 0 if episode_timesteps + 1 == training_config.max_episode_steps else int(done)
        done = 1 if episode_timesteps + 1 == training_config.max_episode_steps else int(done)
        episode_reward += reward

        REPLAY_BUF.add(state, next_state, action, reward, done_bool)

        state = next_state
        episode_timesteps += 1
        timestep_since_eval += 1
        timestep += 1
    evals.append(evaluate(TRAINER, VENV, DEVICE))
    if training_config.save_model_path is not None:
        TRAINER.save("drl_agent", training_config.save_model_path)
    np.save(OUTPUT_DIR + "\\evals", evals)