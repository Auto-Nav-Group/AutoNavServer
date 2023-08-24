from utils import Geometry
import time, json

class NodeGraph:

    def __init__(self):
        self.map = None
        self.nodes = []
        self.edges = []
    def create_from_map(self, mapinp):
        self.map = mapinp
        currentTime = time.time()
        for i in range(len(mapinp.obstacles)):
            pt1 = Geometry.Point(mapinp.obstacles[i].Loc.x - mapinp.robotSize.width / 2, mapinp.obstacles[i].Loc.y - mapinp.robotSize.height / 2)
            pt2 = Geometry.Point(mapinp.obstacles[i].Loc.x + mapinp.robotSize.width / 2 + mapinp.obstacles[i].Size.width, mapinp.obstacles[i].Loc.y - mapinp.robotSize.height / 2)
            pt3 = Geometry.Point(mapinp.obstacles[i].Loc.x - mapinp.robotSize.width / 2, mapinp.obstacles[i].Loc.y + mapinp.robotSize.height / 2 + mapinp.obstacles[i].Size.height)
            pt4 = Geometry.Point(mapinp.obstacles[i].Loc.x + mapinp.robotSize.width / 2 + mapinp.obstacles[i].Size.width, mapinp.obstacles[i].Loc.y + mapinp.robotSize.height / 2 + mapinp.obstacles[i].Size.height)
            if not mapinp.is_outside_map(pt1):
                self.nodes.append(self.Node(pt1))
            if not mapinp.is_outside_map(pt2):
                self.nodes.append(self.Node(pt2))
            if not mapinp.is_outside_map(pt3):
                self.nodes.append(self.Node(pt3))
            if not mapinp.is_outside_map(pt4):
                self.nodes.append(self.Node(pt4))
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if i != j:
                    intersected = False
                    for k in range(len(mapinp.obstacles)):
                        if Geometry.line_intersects_rect(self.nodes[i].Loc, self.nodes[j].Loc, mapinp.obstacles[k].Loc, mapinp.obstacles[k].Size):
                            intersected = True
                    if not intersected:
                        edge = self.Edge(self.nodes[i], self.nodes[j], self.nodes[i].Loc.distance(self.nodes[j].Loc))
                        self.nodes[i].Edges.append(edge)
                        self.nodes[j].Edges.append(edge)
                        self.edges.append(edge)
        print("Created node map of nodes: " + str(len(self.nodes)) + " and edges: " + str(len(self.edges)) + " in " + str(round(time.time()-currentTime, 3)) + " seconds")
    def add_node(self, node):
        self.nodes.append(node)
        for i in range(len(self.nodes)-1):
            intersected = False
            for k in range(len(self.map.obstacles)):
                if Geometry.line_intersects_rect(node.Loc, self.nodes[i].Loc, self.map.obstacles[k].Loc,
                                               self.map.obstacles[k].Size):
                    intersected = True
            if not intersected:
                edge = self.Edge(node, self.nodes[i], node.Loc.distance(self.nodes[i].Loc))
                self.nodes[i].Edges.append(edge)
                node.Edges.append(edge)
                self.edges.append(edge)

    def create_json(self, path):
        towrite = {
            "edges" : [self.edges]
        }

        json_obj = json.dumps(towrite, indent=4, default=lambda o: o.__dict__)

        with open(path, "w") as outfile:
            outfile.write(json_obj)
    def to_json(self):
        edgestrs = []
        for edge in self.edges:
            edgestrs.append(edge.to_json())
        return {
            "edges" : edgestrs
        }

    class Node:
        def __init__(self, location):
            self.Loc = location
            self.Edges = []
        def to_json(self):
            edges = []
            for edge in self.Edges:
                edges.append(edge.to_json())
            dictionary = {
                "Type" : "Node",
                "Location" : self.Loc.unpack(),
                "Edges" : edges
            }
            return json.dumps(dictionary)
    class Edge:
        def __init__(self, node1, node2, weight):
            self.loc1 = node1.Loc
            self.loc2 = node2.Loc
            self.weight = weight
            #TODO: Change start location and end location to endpoints and add a function to get the other point.
        def other_loc(self, loc):
            if loc != self.loc1:
                return self.loc1
            if loc != self.loc2:
                return self.loc2
        def to_json(self):
            return {
                "Type" : "Edge",
                "Location1" : self.loc1.unpack(),
                "Location2" : self.loc2.unpack(),
                "Weight" : self.weight
            }