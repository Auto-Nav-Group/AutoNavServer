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
                        #if Geometry.line_intersects_rect(self.nodes[i].Loc, self.nodes[j].Loc, Geometry.Point(mapinp.obstacles[k].Loc.x-mapinp.robotSize.width/2, mapinp.obstacles[k].Loc.y-mapinp.robotSize.height/2), mapinp.obstacles[k].Size.add(Geometry.Size(mapinp.robotSize.width/2, mapinp.robotSize.height/2))):
                            intersected = True
                    if not intersected:
                        edge = self.Edge(self.nodes[i], self.nodes[j], self.nodes[i].Loc.distance(self.nodes[j].Loc))
                        self.nodes[i].Edges.append(edge)
                        self.nodes[j].Edges.append(edge)
                        self.edges.append(edge)
        #TODO: Add second layer of processing of edges in order to add nodes at intersections of edges.
        newedges = self.edges
        newnodes = self.nodes
        for i in range(len(self.edges)):
            for j in range(i+1, len(self.edges)):
                if i != j:
                    point = Geometry.line_intersects_line(self.edges[i].loc1, self.edges[i].loc2, self.edges[j].loc1, self.edges[j].loc2)
                    if point is not False:
                        if point is not self.edges[i].loc1:
                            newedges.append(self.Edge.from_points(self.edges[i].loc1, point, self.edges[i].loc1.distance(point)))
                        if point is not self.edges[i].loc2:
                            newedges.append(self.Edge.from_points(self.edges[i].loc2, point, self.edges[i].loc2.distance(point)))
                        if point is not self.edges[i].loc1 or point is not self.edges[i].loc2:
                            newedges.pop(i)
                        if point is not self.edges[j].loc1:
                            newedges.append(self.Edge.from_points(self.edges[j].loc1, point, self.edges[j].loc1.distance(point)))
                        if point is not self.edges[j].loc2:
                            newedges.append(self.Edge.from_points(self.edges[j].loc2, point, self.edges[j].loc2.distance(point)))
                        if point is not self.edges[j].loc1 or point is not self.edges[j].loc2:
                            newedges.pop(j)
                        if point not in newnodes:
                            newnodes.append(self.Node(point))
        self.nodes = newnodes
        self.edges = newedges
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
    def to_compact_json(self):
        return json.dumps({
            "Type" : "Node",
            "Location" : self.Loc.unpack()
        })
class Edge:
    def __init__(self, node1, node2, weight):
        self.loc1 = node1.Loc
        self.loc2 = node2.Loc
        self.weight = weight
        #TODO: Change start location and end location to endpoints and add a function to get the other point.

    @staticmethod
    def from_points(loc1, loc2, weight):
        return NodeGraph.Edge(NodeGraph.Node(loc1), NodeGraph.Node(loc2), weight)

    @staticmethod
    def weight_from_points(loc1, loc2):
        return loc1.distance(loc2)

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