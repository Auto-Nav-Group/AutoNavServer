from drl_venv import DRL_VENV
from map import Map
import json

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

DRL_VENV = DRL_VENV(environment_dimensions=[mapobj.size.width, mapobj.size.height], map=mapobj)