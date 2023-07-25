import time
import json

class Point:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
    def distance(self, point):
        return ((self.x-point.x)**2+(self.y-point.y)**2)**0.5
    def unpack(self):
        return (self.x, self.y)

class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def unpack(self):
        return (self.width, self.height)
class Obstacle:
    def __init__(self, Location, Size, Rot):
        self.Loc = Location
        self.Size = Size
        self.Rot = Rot


class Map:
    def __init__(self, json):
        self.size = None
        self.obstacles = []
        self.robotSize = Size(0,0)
        self.json = json
        self.parseJSON(self.json)

    def parseJSON(self, JSON):
        self.size = Size(JSON[0]['sizex'], JSON[0]['sizey'])
        self.robotSize = Size(JSON[0]['robotWidth'], JSON[0]['robotHeight'])
        for i in range(len(JSON[1][0])):
            obj = JSON[1][0][i]
            size = Size(obj['width'], obj['height'])
            self.obstacles.append(Obstacle(Point(obj['locationx'], obj['locationy']), size, obj['rotationangle']))
    def isOutsideMap(self, point):
        if point.x>self.size.width or point.x<0 or point.y>self.size.height or point.y<0:
            return True
        return False
class NodeGraph:

    def __init__(self):
        self.nodes = []
        self.edges = []
    def createFromMap(self, map):
        currentTime = time.time()
        for i in range(len(map.obstacles)):
            pt1 = Point(map.obstacles[i].Loc.x-map.robotSize.width/2, map.obstacles[i].Loc.y-map.robotSize.height/2)
            pt2 = Point(map.obstacles[i].Loc.x+map.robotSize.width/2 + map.obstacles[i].Size.width, map.obstacles[i].Loc.y-map.robotSize.height/2)
            pt3 = Point(map.obstacles[i].Loc.x-map.robotSize.width/2, map.obstacles[i].Loc.y+map.robotSize.height/2 + map.obstacles[i].Size.height)
            pt4 = Point(map.obstacles[i].Loc.x+map.robotSize.width/2 + map.obstacles[i].Size.width, map.obstacles[i].Loc.y+map.robotSize.height/2 + map.obstacles[i].Size.height)
            if not map.isOutsideMap(pt1):
                self.nodes.append(self.Node(pt1))
            if not map.isOutsideMap(pt2):
                self.nodes.append(self.Node(pt2))
            if not map.isOutsideMap(pt3):
                self.nodes.append(self.Node(pt3))
            if not map.isOutsideMap(pt4):
                self.nodes.append(self.Node(pt4))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    intersected = False
                    for k in range(len(map.obstacles)):
                        if LineIntersectsRect(self.nodes[i].Loc, self.nodes[j].Loc, map.obstacles[k].Loc, map.obstacles[k].Size):
                            intersected = True
                    if not intersected:
                        edge = self.Edge(self.nodes[i], self.nodes[j], self.nodes[i].Loc.distance(self.nodes[j].Loc))
                        self.nodes[i].Edges.append(edge)
                        self.nodes[j].Edges.append(edge)# TODO: Edge is double counted
                        self.edges.append(edge)
        print("Created node map of nodes: " + str(len(self.nodes)) + " and edges: " + str(len(self.edges)) + " in " + str(time.time()-currentTime) + " seconds")

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

def LineIntersectsRect(line_start, line_end, rect_position, rect_size):
    def on_segment(p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

    def orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # Collinear points
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def do_segments_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    # Define the rectangle's four corners
    rect_end = Point(rect_position.x + rect_size.width, rect_position.y + rect_size.height)
    top_left = rect_position
    top_right = Point(rect_end.x, rect_position.y)
    bottom_left = Point(rect_position.x, rect_end.y)
    bottom_right = rect_end

    # Check if the line segment intersects with any of the rectangle's edges
    if (do_segments_intersect(line_start, line_end, top_left, top_right) or
        do_segments_intersect(line_start, line_end, top_right, bottom_right) or
        do_segments_intersect(line_start, line_end, bottom_right, bottom_left) or
        do_segments_intersect(line_start, line_end, bottom_left, top_left)):
        return True

    return False