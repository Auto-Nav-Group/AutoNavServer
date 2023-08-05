import math
from nodegraph import NodeGraph
from utils import Geometry

class trajectory_edge:
    @staticmethod
    def from_edge(edge):
        delta = Geometry.Point(edge.loc1.x-edge.loc2.x, edge.loc1.y-edge.loc2.y)
        angle = math.atan(delta.y/delta.x)
        return trajectory_edge(edge.loc1, angle, edge.weight)
    def __init__(self, startpoint, angle, dist, top_speed=0):
        self.startpoint = startpoint
        self.angle = angle
        self.dist = dist
        self.top_speed = top_speed
        self.acceleration_dist = 0
        self.deceleration_dist = 0
        self.full_speed_dist = 0
        self.robot = None
    def calculate_speed_dists(self):
        if self.robot is None:
            return
        if self.top_speed == 0:
            self.top_speed = self.robot.max_speed
        if self.dist >= self.top_speed/self.robot.safe_acceleration_speed+self.top_speed/self.robot.safe_decelleration_speed:
            self.acceleration_dist = self.top_speed/(2*self.robot.safe_acceleration_speed)
            self.deceleration_dist = self.top_speed/(2 * self.robot.safe_decelleration_speed)
            self.full_speed_dist = self.dist-self.acceleration_dist-self.decceleration_dist
        else:
            self.acceleration_dist=self.dist/2
            self.decceleration_dist=self.dist/2
        # TODO: Implement tipping calculations
        '''
        Tipping info:
        Pivot points, calculate the amount of force necessary to rotate the robot.
        I-gt>0 where I is inertial force, g is gravity and t is time to reach tipping angle
        '''

def process_path(path, robot):
    traj_path=[]
    for i in range(len(path)-1):
        traj_path.append(trajectory_edge.from_edge(NodeGraph.Edge.from_two_nodes(path[i], path[i+1])))
        traj_path[i].robot = robot
        traj_path[i].calculate_speed_dists()