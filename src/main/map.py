import json
from utils import Geometry

class Obstacle:
    def __init__(self, location, size, rot):
        self.Loc = location
        self.Size = size
        self.Rot = rot


class Map:
    def __init__(self, jsonfile):
        self.size = None
        self.obstacles = []
        self.vobstacles = []
        self.robotSize = Geometry.Size(0,0)
        self.json = jsonfile
        self.parse_json(self.json)

    def parse_json(self, jsonfile):
        try:
            self.size = Geometry.Size(jsonfile[0]['sizex'], jsonfile[0]['sizey'])
            self.robotSize = Geometry.Size(jsonfile[0]['robotWidth'], jsonfile[0]['robotHeight'])
            for i in range(len(jsonfile[1][0])):
                obj = jsonfile[1][0][i]
                size = Geometry.Size(obj['width'], obj['height'])
                self.obstacles.append(Obstacle(Geometry.Point(obj['locationx'], obj['locationy']), size, obj['rotationangle']))
                self.vobstacles.append(Obstacle(Geometry.Point(obj['locationx']-self.robotSize.width/2, obj['locationy']-self.robotSize.height/2), Geometry.Size(size.width+self.robotSize.width, size.height+self.robotSize.height), obj['rotationangle']))
        except Exception as e:
            print("Error parsing JSON. Error: "+str(e))
    def is_outside_map(self, point):
        if point.x>self.size.width or point.x<0 or point.y>self.size.height or point.y<0:
            return True
        return False
    def to_json(self):
        towrite = {
            "Type" : "Map",
            "JSON" : self.json
        }
        json_obj = json.dumps(towrite, indent=4, default=lambda o: o.__dict__)
        return json_obj