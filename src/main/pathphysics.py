import math
from nodegraph import NodeGraph
from utils import Geometry

class trajectory_edge:
    @staticmethod
    def from_edge(edge):
        delta = Geometry.Point(edge.loc1.x-edge.loc2.x, edge.loc1.y-edge.loc2.y)
        angle = math.atan(delta.y/delta.x)
        return trajectory_edge(edge.loc1, angle, edge.weight)
    def __init__(self, startpoint, angle, dist, speed=0):
        self.startpoint = startpoint
        self.angle = angle
        self.dist = dist
        self.speed = speed

def process_path(path):
    traj_path=[]
    for i in range(len(path)-1):
        traj_path.append(trajectory_edge.from_edge(NodeGraph.Edge.from_two_nodes(path[i], path[i+1])))