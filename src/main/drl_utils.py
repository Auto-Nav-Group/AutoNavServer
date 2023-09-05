import numpy as np
import random

class ObjectState:
    def __init__(self):
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 0]
        self.linear_velocity = [0, 0, 0]
        self.angular_velocity = [0, 0, 0]

class Quaternion:
    def __init__(self):
        self.w = 0
        self.x = 0
        self.y = 0
        self.z = 1
    def define(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_euler(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = Quaternion()

        q.define(cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)

        return q

    def to_euler(self):
        roll = np.arctan2(2*(self.w*self.x+self.y*self.z), 1-2*(self.x**2+self.y**2))
        pitch = np.arcsin(2*(self.w*self.y-self.z*self.x))
        yaw = np.arctan2(2*(self.w*self.z+self.x*self.y), 1-2*(self.y**2+self.z**2))

        return roll, pitch, yaw

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the buffer contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0