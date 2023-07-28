import time
from threading import Thread
from map import Map
from nodegraph import NodeGraph
from pathfinder import PathFinder
from server import run_server
from utils import Geometry
import requests
import json

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'

JSON = json.load(open(path))
map = Map(JSON)

GRAPH = NodeGraph()
GRAPH.createFromMap(map)

GRAPH.createJSON("G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\OUTPUTNODEGRAPH.json")
pathfinder = PathFinder(GRAPH)

pathfinder.DEBUG_benchmarkrecalculate()

server_thread = Thread(target=run_server, args=('localhost', 8000), name='Server')
server_thread.start()

time.sleep(5)
requests.get('http://localhost:8000', headers={'Command' : 'pathfind', 'startnode' : NodeGraph.Node(Geometry.Point(1,1)).toJSON(), 'endnode' : NodeGraph.Node(Geometry.Point(5,5)).toJSON()})