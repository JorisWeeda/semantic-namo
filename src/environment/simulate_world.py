from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import io
import copy
import math
import random
import roslib
import rospy
import torch
import yaml
import zerorpc

import numpy as np

from shapely.geometry import Polygon
from shapely.affinity import rotate

from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler, euler_from_quaternion

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
        self.vel_tolerance = params['controller']['vel_tolerance']
        self.max_f_contact = params['controller']['max_f_contact']

        self._goal = None
        self._mode = None

        self.is_goal_reached = False

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, use_viewer: bool):
        world = cls.create(config, layout, use_viewer)
        world.configure()
        return world

    @classmethod
    def create(cls, config, layout, use_viewer):
        simulation = IsaacGymWrapper(
            config["isaacgym"],
            init_positions=config["initial_actor_positions"],
            actors=config["actors"],
            num_envs=1,
            viewer=use_viewer,
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
    
    def configure(self, additions=None):
        if additions is None:
            if self.params["random"]:
                additions = self.random_additions()
            else:
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

    def random_additions(self):        
        additions = self.create_walls()

        range_x = self.params['range_x']
        range_y = self.params['range_y']

        area = (range_x[1] - range_x[0]) * (range_y[1] - range_y[0])

        stationary_percentage = self.params['stationary_percentage']
        stationary_size_noise = self.params['stationary_size_noise']

        adjustable_percentage = self.params['adjustable_percentage']
        adjustable_size_noise = self.params['adjustable_size_noise']

        inflation = self.params["scheduler"]["path_inflation"]
        init_pose = self.params["environment"]["robot"]["init_state"]
        goal_pose = self.params["goal"]

        excluded_poses = ({'init_pos': init_pose, 'init_ori': [0., 0., 0.], 'size': [inflation, inflation]},
                          {'init_pos': goal_pose, 'init_ori': [0., 0., 0.], 'size': [inflation, inflation]})

        stationary_area_target = area * stationary_percentage
        adjustable_area_target = area * adjustable_percentage

        current_stationary_area = 0.
        current_adjustable_area = 0.

        while current_stationary_area < stationary_area_target:
            obstacle = copy.deepcopy(self.params["objects"]["stationary"])
            obstacle["name"] = f"Obstacle {len(additions)}"

            random_yaw = random.uniform(-np.pi, np.pi)
            random_x = random.uniform(*range_x)
            random_y = random.uniform(*range_y)

            init_pos = [random_x, random_y, 0.5]            
            init_ori = self.yaw_to_quaternion(random_yaw)

            obstacle["init_ori"] = init_ori
            obstacle["init_pos"] = init_pos

            size_x, size_y, size_z = obstacle["size"]
            obstacle["size"][0] = size_x + np.random.uniform(-stationary_size_noise * size_x, stationary_size_noise * size_x)
            obstacle["size"][1] = size_y + np.random.uniform(-stationary_size_noise * size_y, stationary_size_noise * size_y)
            obstacle["size"][2] = size_z + np.random.uniform(-stationary_size_noise * size_z, stationary_size_noise * size_z)

            if not self.is_obstacle_overlapping(init_pos, obstacle["size"], init_ori, additions, excluded_poses):
                current_stationary_area += (obstacle["size"][0] * obstacle["size"][1])
                additions.append(obstacle)

        while current_adjustable_area < adjustable_area_target:
            obstacle = copy.deepcopy(self.params["objects"]["adjustable"])
            obstacle["name"] = f"Obstacle {len(additions)}"

            random_yaw = random.uniform(-np.pi, np.pi)
            random_x = random.uniform(*range_x)
            random_y = random.uniform(*range_y)

            init_pos = [random_x, random_y, 0.5]
            init_ori = self.yaw_to_quaternion(random_yaw)

            obstacle["init_ori"] = init_ori
            obstacle["init_pos"] = init_pos

            size_x, size_y, size_z = obstacle["size"]
            obstacle["size"][0] = size_x + np.random.uniform(-adjustable_size_noise * size_x, adjustable_size_noise * size_x)
            obstacle["size"][1] = size_y + np.random.uniform(-adjustable_size_noise * size_y, adjustable_size_noise * size_y)
            obstacle["size"][2] = size_z + np.random.uniform(-adjustable_size_noise * size_z, adjustable_size_noise * size_z)

            if not self.is_obstacle_overlapping(init_pos, obstacle["size"], init_ori, additions, excluded_poses):
                current_adjustable_area += (obstacle["size"][0] * obstacle["size"][1])
                additions.append(obstacle)

        return additions

    def create_walls(self, thickness=0.01, height=0.5):
        range_x = self.params["range_x"]
        range_y = self.params["range_y"]

        def new_wall(name, size, init_pos, init_ori):
            wall = copy.deepcopy(self.params["objects"]['wall'])
            wall["name"] = name
            wall["size"] = size
            wall["init_pos"] = init_pos
            wall["init_ori"] = self.yaw_to_quaternion(init_ori[-1])
            return wall

        walls = []
    
        walls.append(new_wall("l-demarcation-wall", [range_x[1]-range_x[0], thickness, height], [(range_x[1]+range_x[0])/2, range_y[1]+thickness/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("r-demarcation-wall", [range_x[1]-range_x[0], thickness, height], [(range_x[1]+range_x[0])/2, range_y[0]-thickness/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("f-demarcation-wall", [thickness, range_y[1]-range_y[0], height], [range_x[0]-thickness/2, (range_y[1]+range_y[0])/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("b-demarcation-wall", [thickness, range_y[1]-range_y[0], height], [range_x[1]+thickness/2, (range_y[1]+range_y[0])/2, 0], [0.0, 0.0, 0.0]))

        return walls

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

        tensor_init = self.get_robot_dofs()
        tensor_goal = torch.tensor([goal[0], goal[1], 0., *quaternions])
        tensor_mode = torch.tensor([mode[0], mode[1]])

        rospy.loginfo(f"New starting state: {tensor_init}")
        rospy.loginfo(f"New objective goal: {tensor_goal}")
        rospy.loginfo(f"New objective mode: {tensor_mode}")

        bytes_init = self.torch_to_bytes(tensor_init)
        bytes_goal = self.torch_to_bytes(tensor_goal)
        bytes_mode = self.torch_to_bytes(tensor_mode)

        self.controller.update_objective(bytes_init, bytes_goal, bytes_mode)

    def get_robot_dofs(self):
        return self.simulation.dof_state[0].tolist()
    
    def get_actor_states(self):
        return self.simulation.env_cfg, self.simulation.root_state[0, :, :]

    def get_net_forces(self):
        net_contact_forces = torch.sum(torch.abs(torch.cat((self.simulation.net_contact_force[:, 0].unsqueeze(1), self.simulation.net_contact_force[:, 1].unsqueeze(1)), 1)), 1)
        number_of_bodies = int(net_contact_forces.size(dim=0) / self.simulation.num_envs)

        reshaped_contact_forces = net_contact_forces.reshape([self.simulation.num_envs, number_of_bodies])
        return torch.sum(reshaped_contact_forces, dim=1)[0]

    def get_elapsed_time(self):
        return self.simulation._gym.get_elapsed_time(self.simulation.sim)

    def get_rollout_states(self):
        return self.bytes_to_torch(self.controller.get_states())

    def get_rollout_best_state(self):
        return self.bytes_to_torch(self.controller.get_n_best_samples())

    def destroy(self):
        self.simulation.stop_sim()

    def check_goal_reached(self):
        if self._goal is None:
            return None

        rob_dof = self.get_robot_dofs()
        rob_pos = np.array([rob_dof[0], rob_dof[2]])

        self.is_goal_reached = False
        if torch.linalg.norm(self._goal[:2] - rob_pos) < self.pos_tolerance :
            self.is_goal_reached = True

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw).tolist()

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return euler_from_quaternion(quaternion)[-1]

    @staticmethod
    def is_obstacle_overlapping(new_position, new_size, new_orientation, obstacles, excluded_poses, margin=0.1):
        new_polygon = Polygon([
            (new_position[0] - new_size[0] / 2 - margin, new_position[1] - new_size[1] / 2 - margin),
            (new_position[0] + new_size[0] / 2 + margin, new_position[1] - new_size[1] / 2 - margin),
            (new_position[0] + new_size[0] / 2 + margin, new_position[1] + new_size[1] / 2 + margin),
            (new_position[0] - new_size[0] / 2 - margin, new_position[1] + new_size[1] / 2 + margin)
        ])

        yaw_new_polygon = np.rad2deg(SimulateWorld.quaternion_to_yaw(new_orientation))
        new_polygon = rotate(new_polygon, yaw_new_polygon, origin=new_position)

        all_excluded_poses = list(obstacles) + list(excluded_poses)
        for excluded_pose in all_excluded_poses:
            obstacle_polygon = Polygon([
                (excluded_pose['init_pos'][0] - excluded_pose['size'][0] / 2 - margin,
                excluded_pose['init_pos'][1] - excluded_pose['size'][1] / 2 - margin),
                (excluded_pose['init_pos'][0] + excluded_pose['size'][0] / 2 + margin,
                excluded_pose['init_pos'][1] - excluded_pose['size'][1] / 2 - margin),
                (excluded_pose['init_pos'][0] + excluded_pose['size'][0] / 2 + margin,
                excluded_pose['init_pos'][1] + excluded_pose['size'][1] / 2 + margin),
                (excluded_pose['init_pos'][0] - excluded_pose['size'][0] / 2 - margin,
                excluded_pose['init_pos'][1] + excluded_pose['size'][1] / 2 + margin)
            ])

            yaw_obstacle_polygon = np.rad2deg(SimulateWorld.quaternion_to_yaw(excluded_pose['init_ori']))
            obstacle_polygon = rotate(obstacle_polygon, yaw_obstacle_polygon, origin=excluded_pose['init_pos'])
        
            if new_polygon.intersects(obstacle_polygon):
                return True

        return False

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
    