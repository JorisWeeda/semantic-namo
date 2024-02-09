from motion.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from motion.mppiisaac.utils.config_store import ExampleConfig

import hydra
import zerorpc

from omegaconf import OmegaConf

from namo import Objective


@hydra.main(version_base=None, config_path="config", config_name="config_heijn_push")
def server(cfg: ExampleConfig):
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg["mppi"].device)

    try:
        planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, None))
        planner.bind("tcp://0.0.0.0:4242")
        planner.run()

    except zerorpc.exceptions.LostRemote:
        print("Server disconnected. Exiting client.")


if __name__ == "__main__":
    server()



