import json
from map import Map
from nodegraph import NodeGraph
from pathfinder import PathFinder

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'

JSON = json.load(open(path))
map = Map(JSON)

GRAPH = NodeGraph()
GRAPH.createFromMap(map)

GRAPH.createJSON("G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\OUTPUTNODEGRAPH.json")
pathfinder = PathFinder(GRAPH)

pathfinder.DEBUG_benchmarkrecalculate()