from drl_venv import RobotVEnv
from generate_urdf_file_map import from_map, ASSET_PATH
from drl_training_executor import TrainingExecutor
from map import Map
#from onshape_to_robot import onshape_to_robot
import json
import sys
import time


if sys.platform == "win32":
    path = "/assets/testing/BasicMap.json"
    logger_path = "/output/logs"
elif sys.platform == "linux" or sys.platform == "linux2":
    path = "/home/jovyan/workspace/AutoNavServer/assets/testing/BasicMap.json"
    logger_path = "/home/jovyan/workspace/AutoNavServer/output/logs"
elif sys.platform == "darwin":
    path = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/BasicMap.json"
    logger_path = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/output/logs"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
# path = '/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/testing/FRC2023Map.json'

JSON = json.load(open(path))
mapobj = Map(JSON)

#onshape_to_robot.

from_map(mapobj)
DRL_VENV = RobotVEnv(map=mapobj, assets_path=ASSET_PATH)

#config = training_config(save_model_path="G:\\Projects\\AutoNav\\AutoNavServer\\output\\drl\\model", batch_size=40, discount=0.999999, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, exploration_noise=1, exploration_min=0.1, exploration_decay_steps=500000, max_episode_steps=1000)
#start_training_process(config, DRL_VENV)

#train(DRL_VENV)

te = TrainingExecutor(mapobj, logger_path)
#te.train(DRL_VENV, plotter_display=False)
te.train(DRL_VENV, plotter_display=False)
"""while True:
    inp = input("T - Train\n"
          "L - Load\n"
                "R - Run\n")
    inp = inp.upper()
    if inp == "T":
        #train(DRL_VENV)

        te.train(DRL_VENV, plotter_display=False)
    elif inp == "L":
        e = te.load()
        te.train(DRL_VENV, start_ts=e, plotter_display=False)
    elif inp == "R":
        te.train(DRL_VENV, test=True)
    time.sleep(1)"""