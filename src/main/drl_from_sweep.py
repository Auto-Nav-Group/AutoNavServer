from drl_networks_ddpg import TrainingExecutor
from generate_urdf_file_map import from_map, ASSET_PATH
from drl_venv import DRL_VENV
from map import Map
import wandb
import json

wandb.init(project="autonav")

LEN = 4000

path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\BasicMap.json'
# path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = DRL_VENV(map=mapobj, assets_path=ASSET_PATH)


config = wandb.config
wandb.run.name = config.name

TrainingExecutor = TrainingExecutor(mapobj)

TrainingExecutor.train(DRL_VENV, config=config, num_episodes=LEN, map = mapobj, plotter_display=False)