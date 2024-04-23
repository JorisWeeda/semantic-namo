from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import io
import math
import random
import roslib
import rospy
import torch
import yaml
import zerorpc

import numpy as np

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

        self.pos_tolerance = params['controller']['pos_tolerance']
        self.yaw_tolerance = params['controller']['yaw_tolerance']

        self._goal = None
        self._mode = None

        self.is_goal_reached = False

    @classmethod
    def build(cls, config: ExampleConfig, layout: str):
        world = cls.create(config, layout)
        world.configure()
        return world

    @classmethod
    def create(cls, config, layout):
        simulation = IsaacGymWrapper(
            config["isaacgym"],
            init_positions=config["initial_actor_positions"],
            actors=config["actors"],
            num_envs=1,
            viewer=True,
            device=config["mppi"].device,
        )

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}

        controller = zerorpc.Client()
        controller.connect("tcp://127.0.0.1:4242")

        return cls(params, config, simulation, controller)
    
    def configure(self):
        additions = self.create_additions()

        self.simulation.add_to_envs(additions)
        self.controller.add_to_env(additions)

        init_state = self.params["environment"]["robot"]["init_state"]
        (x, y, yaw) = init_state[0], init_state[1], init_state[2] * (math.pi / 180.0)

        self.simulation.set_actor_dof_state(torch.tensor([x, 0., y, 0., yaw, 0.]))
        self.update_objective((x, y, yaw), (0., 0.))

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation._gym, self.simulation.viewer, cam_pos, cam_tar)

    def create_additions(self):
        additions =[]

        range_x = self.params['range_x']
        range_y = self.params['range_y']

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

                init_ori = obstacle.get("init_ori", None)
                init_pos = obstacle.get("init_pos", None)

                random_yaw = random.uniform(0, 360)
                random_x = random.uniform(*range_x)
                random_y = random.uniform(*range_y)

                init_ori = init_ori if init_ori else [0., 0. , random_yaw]
                init_pos = init_pos if init_pos else [random_x, random_y, 0.5]

                init_ori = Rotation.from_euler('xyz', init_ori, degrees=True).as_quat().tolist()
                
                obstacle["init_ori"] = init_ori
                obstacle["init_pos"] = init_pos

                additions.append(obstacle)

        return additions
    
    def run(self):
        df_state_tensor = self.torch_to_bytes(self.simulation.dof_state)
        rt_state_tensor = self.torch_to_bytes(self.simulation.root_state)
        rb_state_tensor = self.torch_to_bytes(self.simulation.rigid_body_state)

        bytes_action = self.controller.compute_action_tensor(df_state_tensor, rt_state_tensor, rb_state_tensor)
        action = self.bytes_to_torch(bytes_action)

        if torch.any(torch.isnan(action)):
            action = torch.zeros_like(action)

        self.check_goal_reached()
        if self.is_goal_reached:
            rospy.loginfo_throttle(1, 'The goal is reached, no action applied to the robot')

        self.simulation.apply_robot_cmd(action)        
        self.simulation.step()

        return action

    def update_objective(self, goal, mode=(0, 0)):
        self._goal = goal
        self._mode = mode

        quaternions = self.yaw_to_quaternion(goal[2])

        tensor_init = self.simulation.dof_state[0]
        tensor_goal = torch.tensor([goal[0], goal[1], 0., *quaternions])
        tensor_mode = torch.tensor([mode[0], mode[1]])

        rospy.loginfo(f"New objective init: {tensor_init}")
        rospy.loginfo(f"New objective goal: {tensor_goal}")
        rospy.loginfo(f"New objective mode: {tensor_mode}")

        bytes_init = self.torch_to_bytes(tensor_init)
        bytes_goal = self.torch_to_bytes(tensor_goal)
        bytes_mode = self.torch_to_bytes(tensor_mode)

        self.controller.update_objective(bytes_init, bytes_goal, bytes_mode)

    def get_elapsed_time(self):
        return self.simulation._gym.get_elapsed_time(self.simulation.sim)

    def get_rollouts(self):
        return self.bytes_to_torch(self.controller.get_rollouts())

    def get_states(self):
        return self.bytes_to_torch(self.controller.get_states())

    def get_best_state(self):
        return self.bytes_to_torch(self.controller.get_n_best_samples())

    def destroy(self):
        self.simulation.stop_sim()

    def check_goal_reached(self):
        if self._goal is None:
            return None

        rob_dof = self.simulation.dof_state[0]
        
        rob_pos = np.array([rob_dof[0], rob_dof[2]])
        rob_yaw = np.array([rob_dof[4]])
        
        distance = np.linalg.norm(rob_pos - self._goal[:2])
        rotation = np.abs(rob_yaw - self._goal[2])

        self.is_goal_reached = False
        if distance < self.pos_tolerance and rotation < self.yaw_tolerance:
            self.is_goal_reached = True

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))

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
    
    @property
    def goal(self):
        return self._goal
    
    @property
    def mode(self):
        return self._mode
    