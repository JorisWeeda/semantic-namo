from control.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import hydra
import io
import math
import roslib
import torch
import yaml
import zerorpc

from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler

from isaacgym import gymapi


class SimulateWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation, controller):
        self.simulation = simulation
        self.controller = controller

        self.params = params
        self.config = config

    @hydra.main(version_base=None, config_path="config")
    def build(config: ExampleConfig):
        return SimulateWorld.create(config)

    @classmethod
    def create(cls, config):
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

        world_config = f'{cls.PKG_PATH}/config/worlds/{config["world"]}.yaml'
        with open(world_config, "r") as stream:
            params = yaml.safe_load(stream)

        controller = zerorpc.Client()
        controller.connect("tcp://127.0.0.1:4242")

        return cls(params, config, simulation, controller)
    
    def configure(self):
        additions = self.create_additions()

        self.simulation.add_to_envs(additions)
        self.controller.add_to_env(additions)

        init_state = self.params["environment"]["robot"]["init_state"]
        (x, y, yaw) = init_state[0], init_state[1], self.degrees_to_radians(init_state[1])

        self.simulation.set_dof_state_tensor(torch.tensor([x, 0., y, 0., yaw, 0.]))
        self.update_objective((x, y, yaw), (0., 0.))

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation.gym, self.simulation.viewer, cam_pos, cam_tar)

    def create_additions(self):
        additions =[]

        if self.params["environment"].get("demarcation", None):
            for wall in self.params["environment"]["demarcation"]:
                obs_type = next(iter(wall))
                obs_args = self.params["objects"][obs_type]

                obstacle = {**obs_args, **wall[obs_type]}

                rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
                obstacle["init_ori"] = list(rot)

                additions.append(obstacle)

        if self.params["environment"].get("obstacles", None):
            for obstacle in self.params["environment"]["obstacles"]:
                obs_type = next(iter(obstacle))
                obs_args = self.params["objects"][obs_type]

                obstacle = {**obs_args, **obstacle[obs_type]}

                rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
                obstacle["init_ori"] = list(rot)

                additions.append(obstacle)

        return additions
    
    def run(self):
        df_state_tensor = self.torch_to_bytes(self.simulation.dof_state[0])
        rt_state_tensor = self.torch_to_bytes(self.simulation.root_state[0])
        rb_state_tensor = self.torch_to_bytes(self.simulation.rigid_body_state[0])

        self.controller.reset_rollout_sim(df_state_tensor, rt_state_tensor, rb_state_tensor)

        action = self.bytes_to_torch(self.controller.command())

        if torch.any(torch.isnan(action)):
            action = torch.zeros_like(action)

        self.simulation.set_dof_velocity_target_tensor(action)
        self.simulation.step()

    def update_objective(self, goal, mode):
        quaternions = self.yaw_to_quaternion(goal[2])

        tensor_goal = torch.tensor([goal[0], goal[1], 0., *quaternions],)
        tensor_mode = torch.tensor([mode[0], mode[1]])
        
        self.controller.update_objective_goal(self.torch_to_bytes(tensor_goal))
        self.controller.update_objective_mode(self.torch_to_bytes(tensor_mode))

    def get_elapsed_time(self):
        return self.simulation.gym.get_elapsed_time(self.simulation.sim)

    def destroy(self):
        self.simulation.gym.destroy_viewer(self.simulation.viewer)
        self.simulation.gym.destroy_sim(self.simulation.sim)

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))

    @staticmethod
    def degrees_to_radians(degrees):
        return degrees * (math.pi / 180.0)

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw)

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
