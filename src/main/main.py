import time
from map import Map
from nodegraph import NodeGraph
from pathfinder import PathFinder
from utils import Geometry
import requests
import json

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

GRAPH = NodeGraph()
GRAPH.create_from_map(mapobj)

GRAPH.create_json("G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\OUTPUTNODEGRAPH.json")
pathfinder = PathFinder(GRAPH)

pathfinder.debug_benchmark_recalculate()

#server_thread = Thread(target=server.start_server, name='Server')
#server_thread.start()

time.sleep(1)
requests.post('http://localhost:8000', headers={'Command' : 'update_map'}, data=mapobj.to_json())
requests.get('http://localhost:8000', headers={'Command' : 'pathfind', 'startnode' : NodeGraph.Node(Geometry.Point(1,1)).to_json(), 'endnode' : NodeGraph.Node(Geometry.Point(5, 5)).to_json()})
requests.post('http://localhost:8000', headers={'Command' : 'kill_thread'})