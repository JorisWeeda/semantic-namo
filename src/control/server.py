from motion.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from motion.mppiisaac.utils.config_store import ExampleConfig

import hydra
import zerorpc

from omegaconf import OmegaConf

from namo import Objective


CONFIG_NAME = "config_dingo_push"


@hydra.main(version_base=None, config_path="config", config_name="config_dingo_push")
def server(cfg: ExampleConfig):
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg["mppi"].device)

    try:
        planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, None))
        planner.bind("tcp://0.0.0.0:4242")
        planner.run()

    except zerorpc.exceptions.LostRemote:
        print("Server disconnected. Exiting client.")

def ros_main():
    hydra.initialize(config_path="../../config", version_base=None)
    
    cfg = hydra.compose(config_name="config_dingo_push")

    server(cfg)