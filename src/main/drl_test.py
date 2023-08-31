from drl_venv import DRL_VENV
from generate_urdf_file_map import from_map, ASSET_PATH
from map import Map
#from onshape_to_robot import onshape_to_robot
import json
import time

# path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\FRC2023Map.json'
path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = DRL_VENV(environment_dimensions=[mapobj.size.width, mapobj.size.height], map=mapobj, assets_path=ASSET_PATH)

while DRL_VENV.pybullet_instance.isConnected():
    DRL_VENV.step(None)
    #time.sleep(1)