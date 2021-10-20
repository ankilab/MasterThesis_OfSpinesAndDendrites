import os
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
# from utils.logging import get_logger
import yaml

from run import run

# logger = get_logger()

ex = Experiment()
# ex.logger = logger
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.automain
def main():
    # Setting the random seed throughout the modules
    # np.random.seed(config["seed"])
    # th.manual_seed(config["seed"])
    # params = deepcopy(sys.argv)
    # th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "config.yaml error: {}".format(exc)

    # Result path
    n = 1
    file_obs_path = os.path.join(results_path, f"{config_dict['algorithm']}_{n}")
    while os.path.isdir(file_obs_path):
        n +=1
        file_obs_path = os.path.join(results_path, f"{config_dict['algorithm']}_{n}")
    os.makedirs(file_obs_path)
    config_dict['result_path'] = file_obs_path

    # now add all the config to sacred
    ex.add_config(config_dict)

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(MongoObserver())
    # ex.observers.append(MongoObserver())

    # run the framework
    run(config=config_dict)






