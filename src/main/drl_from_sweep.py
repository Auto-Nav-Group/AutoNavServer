import sys

from drl_training_executor import TrainingExecutor
from generate_urdf_file_map import from_map, ASSET_PATH
from drl_venv import RobotVEnv#DRL_VENV
from map import Map
import wandb
import json

wandb.init(project="autonav")

LEN = 150000

if sys.platform == "win32":
    path = 'G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\BasicMap.json'
    logger_path = "G:\Projects\AutoNav\AutoNavServer\output\logs"
elif sys.platform == "linux" or sys.platform == "linux2":
    path = '/home/jovyan/workspace/AutoNavServer/assets/testing/FRC2023Map.json'
    logger_path = '/home/jovyan/workspace/AutoNavServer/output/logs'
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
# path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = RobotVEnv(map=mapobj, assets_path=ASSET_PATH)


config = wandb.config

TrainingExecutor = TrainingExecutor(mapobj, config=config, logger_path=logger_path)

TrainingExecutor.train(DRL_VENV, config=config, total_ts=LEN, inpmap= mapobj, plotter_display=False)
