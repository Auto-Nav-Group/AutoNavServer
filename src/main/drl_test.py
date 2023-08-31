from drl_venv import DRL_VENV
from generate_urdf_file_map import from_map, ASSET_PATH
from map import Map
from onshape_to_robot import onshape_to_robot
import json

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = DRL_VENV(environment_dimensions=[mapobj.size.width, mapobj.size.height], map=mapobj, assets_path=ASSET_PATH)