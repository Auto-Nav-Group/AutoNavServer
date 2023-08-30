import numpy as np

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

    def from_euler(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        self.w = cy * cp * cr + sy * sp * sr
        self.x = cy * cp * sr - sy * sp * cr
        self.y = sy * cp * sr + cy * sp * cr
        self.z = sy * cp * cr - cy * sp * sr