from drl_venv import DRL_VENV
from generate_urdf_file_map import from_map, ASSET_PATH
from drl_pathfinding import training_config, start_training_process
from drl_network_simple import train, train_load
from map import Map
#from onshape_to_robot import onshape_to_robot
import json
import time

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\BasicMap.json'
# path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = DRL_VENV(map=mapobj, assets_path=ASSET_PATH)

#config = training_config(save_model_path="G:\\Projects\\AutoNav\\AutoNavServer\\output\\drl\\model", batch_size=40, discount=0.999999, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, exploration_noise=1, exploration_min=0.1, exploration_decay_steps=500000, max_episode_steps=1000)
#start_training_process(config, DRL_VENV)

#train(DRL_VENV)

while True:
    inp = input("T - Train\n"
          "L - Load")
    inp = inp.upper()
    if inp == "T":
        train(DRL_VENV)
    elif inp == "L":
        train_load(DRL_VENV)
    time.sleep(1)