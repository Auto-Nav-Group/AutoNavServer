from utils import Geometry
import time, json

class NodeGraph:

    def __init__(self):
        self.map = None
        self.nodes = []
        self.edges = []
    def createFromMap(self, map):
        self.map = map
        currentTime = time.time()
        for i in range(len(map.obstacles)):
            pt1 = Geometry.Point(map.obstacles[i].Loc.x-map.robotSize.width/2, map.obstacles[i].Loc.y-map.robotSize.height/2)
            pt2 = Geometry.Point(map.obstacles[i].Loc.x+map.robotSize.width/2 + map.obstacles[i].Size.width, map.obstacles[i].Loc.y-map.robotSize.height/2)
            pt3 = Geometry.Point(map.obstacles[i].Loc.x-map.robotSize.width/2, map.obstacles[i].Loc.y+map.robotSize.height/2 + map.obstacles[i].Size.height)
            pt4 = Geometry.Point(map.obstacles[i].Loc.x+map.robotSize.width/2 + map.obstacles[i].Size.width, map.obstacles[i].Loc.y+map.robotSize.height/2 + map.obstacles[i].Size.height)
            if not map.isOutsideMap(pt1):
                self.nodes.append(self.Node(pt1))
            if not map.isOutsideMap(pt2):
                self.nodes.append(self.Node(pt2))
            if not map.isOutsideMap(pt3):
                self.nodes.append(self.Node(pt3))
            if not map.isOutsideMap(pt4):
                self.nodes.append(self.Node(pt4))
        nodeslen = len(self.nodes)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if i != j:
                    intersected = False
                    for k in range(len(map.obstacles)):
                        if Geometry.LineIntersectsRect(self.nodes[i].Loc, self.nodes[j].Loc, map.obstacles[k].Loc, map.obstacles[k].Size):
                            intersected = True
                    if not intersected:
                        edge = self.Edge(self.nodes[i], self.nodes[j], self.nodes[i].Loc.distance(self.nodes[j].Loc))
                        self.nodes[i].Edges.append(edge)
                        self.nodes[j].Edges.append(edge)
                        self.edges.append(edge)
        print("Created node map of nodes: " + str(len(self.nodes)) + " and edges: " + str(len(self.edges)) + " in " + str(round(time.time()-currentTime, 3)) + " seconds")
    def addNode(self, node):
        self.nodes.append(node)
        for i in range(len(self.nodes)-1):
            intersected = False
            for k in range(len(self.map.obstacles)):
                if Geometry.LineIntersectsRect(node, self.nodes[i].Loc, self.map.obstacles[k].Loc,
                                               self.map.obstacles[k].Size):
                    intersected = True
            if not intersected:
                edge = self.Edge(node, self.nodes[i].Loc, node.Loc.distance(self.nodes[i].Loc))
                self.nodes[i].Edges.append(edge)
                node.Edges.append(edge)
                self.edges.append(edge)

    def createJSON(self, path):
        towrite = {
            "edges" : [self.edges]
        }

        json_obj = json.dumps(towrite, indent=4, default=lambda o: o.__dict__)

        with open(path, "w") as outfile:
            outfile.write(json_obj)

    class Node:
        def __init__(self, Location):
            self.Loc = Location
            self.Edges = []
    class Edge:
        def __init__(self, node1, node2, weight):
            self.startloc = node1.Loc
            self.endloc = node2.Loc
            self.weight = weight
