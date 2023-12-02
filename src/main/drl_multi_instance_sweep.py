from drl_from_sweep import start
from threading import Thread
import sys
import os
import shutil

if sys.platform == "win32":
    wandb_path = 'G:\\Projects\\AutoNav\\AutoNavServer\\wandb'
elif sys.platform == "linux" or sys.platform == "linux2":
    wandb_path = '/home/jovyan/workspace/AutoNavServer/wandb'
elif sys.platform == "darwin":
    wandb_path = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/wandb"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()

NUM_INSTANCES = 2
threads = []

for i in range(NUM_INSTANCES):
    t = Thread(target=start)
    threads.append(t)
    t.start()

for t in threads:
    if t.is_alive():
        t.join()

print("All threads finished")

try:
    for filename in os.listdir(wandb_path):
        filepath = os.path.join(wandb_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
except Exception as e:
    print("Failed to remove wandb folders: " + str(e))