class Size:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
class Obstacle:
    def __init__(self, X, Y, Size, Rot):
        self.X = X
        self.Y = Y
        self.Size = Size
        self.Rot = Rot


class Map:
    def __init__(self, json):
        self.size = None
        self.obstacles = []
        self.json = json
        self.parseJSON(self.json)

    def parseJSON(self, JSON):
        self.size = Size(JSON[0]['sizex'], JSON[0]['sizey'])
        for i in range(len(JSON[1][0])):
            obj = JSON[1][0][i]
            size = Size(obj['width'], obj['height'])
            self.obstacles.append(Obstacle(obj['locationx'], obj['locationy'], size, obj['rotationangle']))