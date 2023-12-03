import numpy as np


class RewardFunction(object):
    def __init__(self):
        pass
    def reset(self):
        pass
    def get_reward(self, *args):
        pass

class ProgressiveRewards(RewardFunction):
    def __init__(self, angle_thresh, angle_decay, speed_weight, angle_speed_weight, hdg_weight, time_weight, closer_weight, closer_o_weight):
        self.hdg_function = lambda x: 1/(angle_thresh*np.sqrt(2*np.pi)) * math.exp(-(x ** 2 / (2 * angle_thresh) ** 2))
        self.hdg_decay_function = lambda x: angle_decay**(angle_thresh*x)
        self.v_weight = speed_weight
        self.anglev_weight = angle_speed_weight
        self.hdg_weight = hdg_weight
        self.time_weight = time_weight
        self.closer_weight = closer_weight
        self.closer_o_weight = closer_o_weight
        self.closest = float('Inf')
        self.closest_o = float('Inf')
    def reset(self):
        self.closest = float('Inf')
        self.closest_o = float('Inf')


    def get_reward(self, distance, min_dist, angle, avel, vel, time):
        hdg_reward = self.hdg_function(angle/np.pi)*self.hdg_weight*self.hdg_decay_function(time)
        vel_reward = self.v_weight*(1-vel)
        angle_vel_reward = self.anglev_weight*abs(avel)
        time_reward = self.time_weight
        close_reward = 0
        close_o_reward = 0
        if self.closest == float('Inf'):
            self.closest = distance
            self.closest_o = min_dist
        else:
            if distance < self.closest:
                if distance != 0:
                    r = self.closest/distance
                else:
                    r = 10
                close_reward = self.closer_weight*r
                self.closest = distance
            if min_dist < self.closest_o:
                if min_dist != 0:
                    r = self.closest_o/min_dist
                else:
                    r = 10
                close_o_reward = self.closer_o_weight*r
                self.closest_o = min_dist
        return hdg_reward + vel_reward + angle_vel_reward + time_reward + close_reward + close_o_reward, vel_reward.item(), angle_vel_reward.item(), time_reward, hdg_reward, close_o_reward, close_reward


class SimpleReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.prev_distance = 0

    def reset(self):
        self.prev_distance = 0

    def get_reward(self, distance):
        reward = self.prev_distance - distance
        self.prev_distance = distance
        return reward