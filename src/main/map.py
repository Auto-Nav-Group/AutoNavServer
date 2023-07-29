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
        try:
            self.size = Size(JSON[0]['sizex'], JSON[0]['sizey'])
            self.robotSize = Size(JSON[0]['robotWidth'], JSON[0]['robotHeight'])
            for i in range(len(JSON[1][0])):
                obj = JSON[1][0][i]
                size = Size(obj['width'], obj['height'])
                self.obstacles.append(Obstacle(Point(obj['locationx'], obj['locationy']), size, obj['rotationangle']))
        except:
            print("Error parsing JSON")
    def isOutsideMap(self, point):
        if point.x>self.size.width or point.x<0 or point.y>self.size.height or point.y<0:
            return True
        return False
    def toJSON(self):
        towrite = {
            "Type" : "Map",
            "JSON" : self.json
        }
        json_obj = json.dumps(towrite, indent=4, default=lambda o: o.__dict__)
        return json_obj
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