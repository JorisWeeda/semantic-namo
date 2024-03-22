from control.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore

import io
import torch
import yaml
import roslib

from scipy.spatial.transform import Rotation

from isaacgym import gymapi


class SimulateWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation):
        self.params = params
        self.config = config

        self.simulation = simulation

    @classmethod
    def create_world(cls, config, world):
        actors=[]
        for actor_name in config["actors"]:
            with open(f'{cls.PKG_PATH}/config/actors/{actor_name}.yaml') as f:
                actors.append(ActorWrapper(**yaml.load(f, Loader=yaml.SafeLoader)))

        simulation = IsaacGymWrapper(
            config["isaacgym"],
            init_positions=config["initial_actor_positions"],
            actors=actors,
            num_envs=1,
            viewer=True,
            device=config["mppi"].device,
        )

        world_config = f'{cls.PKG_PATH}/config/worlds/{world}.yaml'
        with open(world_config, "r") as stream:
            params = yaml.safe_load(stream)

        return cls(params, config, simulation)
    
    def configure_world(self):
        additions = self.create_additions()
        self.simulation.add_to_envs(additions)

        init_pos = self.params["environment"]["robot"]["init_pos"]
        init_vel = self.params["environment"]["robot"]["init_vel"]

        self.simulation.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2]], device=self.simulation.device))

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]

        self.set_viewer(self.simulation.gym, self.simulation.viewer, cam_pos, cam_tar)

    def create_additions(self):
        additions =[]
        
        for wall in self.params["environment"]["demarcation"]:
            obs_type = next(iter(wall))
            obs_args = self.params["objects"][obs_type]

            obstacle = {**obs_args, **wall[obs_type]}
            
            rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
            obstacle["init_ori"] = list(rot)

            additions.append(obstacle)

        for obstacle in self.params["environment"]["obstacles"]:
            obs_type = next(iter(obstacle))
            obs_args = self.params["objects"][obs_type]

            obstacle = {**obs_args, **obstacle[obs_type]}

            rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
            obstacle["init_ori"] = list(rot)

            additions.append(obstacle)

        return additions

    def destroy_viewer(self):
        self.simulation.sim.gym.destroy_viewer(self.simulation.sim.viewer)

    def destroy_sim(self):
        self.simulation.sim.gym.destroy_sim(self.simulation.sim)

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))

    @staticmethod
    def torch_to_bytes(torch_tensor) :
        buff = io.BytesIO()
        torch.save(torch_tensor, buff)
        buff.seek(0)
        return buff.read()

    @staticmethod
    def bytes_to_torch(buffer):
        buff = io.BytesIO(buffer)
        return torch.load(buff)