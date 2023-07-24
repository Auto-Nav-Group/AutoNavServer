import json
from maphandler import Map

path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
maphandler = Map(JSON)

