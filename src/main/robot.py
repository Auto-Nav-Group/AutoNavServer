from utils import Geometry



class Robot:
    def __init__(self, sizex, sizey):
        self.sizex = sizex
        self.sizey = sizey
        self.com=Geometry.Point(sizex/2, sizey/2)
        self.com_height = 0
        self.mass = 0
        self.max_speed = 0
        self.safe_acceleration_speed = 0 # TODO: Auto calculate value as show in pathphysics.py
        self.safe_decelleration_speed = 0