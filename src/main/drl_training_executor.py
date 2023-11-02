import wandb
import numpy as np
import math
import keyboard
import torch
from drl_utils import ReplayMemory, Model_Plotter, Model_Visualizer, Minima_Visualizer, EGreedyNoise, ProgressiveRewards
from drl_networks import SAC, TD3, DEVICE
from logger import logger


TOTAL_TIMESTEPS = 1000000
MAX_TIMESTEP = 100
BATCH_SIZE = 50
SAVE_FREQ = 999
EVAL_FREQ = 250
POLICY_FREQ = 12
VISUALIZER_ENABLED = False
OPTIMIZE = True

STATE_DIM = 8
ACTION_DIM = 2

DEBUG_SAME_SITUATION = False
DEBUG_CIRCLE = False
DEBUG_CRITIC = False

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
ANGLE_SPEED_WEIGHT = -0.45#-0.5
MIN_DIST_WEIGHT = 0
WALL_DIST = 0
ANGLE_DECAY = 1
CLOSER_WEIGHT = 2
CLOSER_O_WEIGHT = -1
hdg_function = lambda x: 1/(ANGLE_THRESH*np.sqrt(2*np.pi)) * math.exp(-(x ** 2 / (2 * ANGLE_THRESH) ** 2))
hdg_decay_function = lambda x: ANGLE_DECAY**(ANGLE_THRESH*x)

START_NOISE = 1
END_NOISE = 0.1
NOISE_DECAY_STEPS = 20000

BASE_LOGGER_CONFIG = {
    "batch_size": BATCH_SIZE,
    "actor_loss": 0,
    "critic_loss": 0,
    "avg_reward" : 0,
    "achieve_rate" : 0,
    "loss_rate" : 0,
    "anglevel_reward" : 0,
    "vel_reward" : 0,
    "start_noise": START_NOISE,
    "end_noise": END_NOISE,
    "noise_decay_steps": NOISE_DECAY_STEPS,
}

EVAL_CONFIG = {
    "eval_reward" : 0,
    "eval_achieve_rate" : 0
}



class TrainingExecutor:
    def __init__(self, inpmap, logger_path, config=None):
        self.config = config
        if self.config is not None:
            if self.config.ntype == "SAC":
                self.network = SAC(STATE_DIM, ACTION_DIM, DEVICE, inpmap, BATCH_SIZE, config=config)
            elif self.config.ntype == "TD3":
                self.network = TD3(STATE_DIM, ACTION_DIM, DEVICE, inpmap, BATCH_SIZE, config=config)
        else:
            self.network = TD3(STATE_DIM, ACTION_DIM, DEVICE, inpmap, BATCH_SIZE)
        self.plotter = None
        self.logger = None
        self.sys_logs = logger("training.log", logger_path)

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
        if config is None:
            config = self.config
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
                self.network.load_model()
            except Exception as e:
                self.sys_logs.log("Failed to load model. Exception: " + str(e), logtype="e")
                return
        else:
            logger_config = dict(BASE_LOGGER_CONFIG, **self.network.get_log_info())
            if EVAL_FREQ != -1:
                logger_config.update(EVAL_CONFIG)
            self.logger = wandb.init(project="autonav", config=logger_config, name="cuda-v1 SAC test")
            wandb.watch(self.network.actor, log='all', log_freq=10)
            #wandb.watch(self.network.critic, log='all', log_freq=10)
        if config is None:
            start_noise = START_NOISE
            noise_decay_steps = NOISE_DECAY_STEPS
            time_weight = TIME_WEIGHT
            angle_speed_weight = ANGLE_SPEED_WEIGHT
            closer_weight = CLOSER_WEIGHT
            closer_o_weight = CLOSER_O_WEIGHT
        else:
            wandb.run.name = config.name + config.ntype
            start_noise = config.start_noise
            noise_decay_steps = config.noise_decay_steps
            time_weight = config.time_weight
            angle_speed_weight = config.angle_speed_weight
            closer_weight = config.closer_weight
            closer_o_weight = config.closer_o_weight
        #noise = OUNoise(ACTION_DIM, max_sigma=start_noise, min_sigma=END_NOISE, decay_period=noise_decay_steps)
        noise = EGreedyNoise(ACTION_DIM, max_sigma=start_noise, min_sigma=END_NOISE, decay_period=noise_decay_steps)
        if self.plotter is None and not test:
            self.plotter = Model_Plotter(total_ts, plotter_display, self.network.mem)
        visualizer = None
        if VISUALIZER_ENABLED or test:
            visualizer = Model_Visualizer(env.basis.size.width, env.basis.size.height)
            keyboard.add_hotkey("ctrl+q", lambda: visualizer.toggle())
        rewardfunc = ProgressiveRewards(ANGLE_THRESH, ANGLE_DECAY, SPEED_WEIGHT, angle_speed_weight, ANGLE_WEIGHT, time_weight, closer_weight, closer_o_weight)
        if EVAL_FREQ != -1:
            self.network.create_eval_set(env)
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
            self.network.update_parameters(total, total, mem=pretrain_mem)
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
                action = self.network.get_action_with_noise(nstate, noise)
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
                        c_loss, a_loss = self.network.update_parameters(ep_steps)
                    if SAVE_FREQ != -1 and (eps+1) % SAVE_FREQ == 0:
                        self.network.save_model()
                        #self.save(timestep)
                    if EVAL_FREQ != -1 and (eps+1) % EVAL_FREQ == 0:
                        eval_rew, eval_ac = self.network.evaluate(env, rewardfunc.get_reward, rewardfunc.reset)
                    else:
                        eval_rew = -1
                        eval_ac = -1
                    self.plotter.update(eps, initdist, episode_reward, episode_vw, episode_avw, episode_tw,
                                        episode_achieve, episode_collide, c_loss,
                                        a_loss, eval_rew, eval_ac, episode_caw, episode_cw)
                print("Episode: " + str(eps) + " Reward: " + str(episode_reward) + " Training completion: " + str(timestep/total_ts*100) + "% through training")
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
                    '''
                    states = torch.stack(states).to(self.network.device)
                    actions = torch.stack(actions).to(self.network.device)
                    action_q = self.network.critic.forward((states, actions))
                    visualizer.clear()
                    visualizer.update(episode_x, episode_y, action_q)
                    visualizer.start(state[2], state[3], state[4], state[5]'''
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