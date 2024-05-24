from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import rospy
import roslib
import torch
import yaml

import numpy as np

from functools import partial
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped

from isaacgym import gymapi


class Viewer:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation):
        self.simulation = simulation

        self.params = params
        self.config = config

        self.obstacle_states = []

        self.robot_prev_msg = None
        self.robot_q_dot = torch.zeros(3)
        self.robot_q = torch.zeros(3)

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, robot_name: str):
        viewer = cls.create(config, layout)
        viewer.configure(robot_name)
        return viewer

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

        return cls(params, config, simulation)
    
    def configure(self, robot_name):
        additions = self.create_additions()
        self.simulation.add_to_envs(additions)

        rospy.Subscriber(f'/vicon/{robot_name}', PoseWithCovarianceStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseWithCovarianceStamped, timeout=10)

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation._gym, self.simulation.viewer, cam_pos, cam_tar)

    def create_additions(self):
        additions =[]

        if self.params.get("environment", None):
            if self.params["environment"].get("obstacles", None):
                for idx, obstacle in enumerate(self.params["environment"]["obstacles"]):
                    obs_type = next(iter(obstacle))
                    obs_args = self.params["objects"][obs_type]

                    obstacle = {**obs_args, **obstacle[obs_type]}
                    topic_name = obstacle.get("topic_name", None)

                    empty_msg = PoseWithCovarianceStamped()
                    pos, ori = empty_msg.pose.pose.position, empty_msg.pose.pose.orientation

                    self.obstacle_states.append([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])

                    rospy.Subscriber(f'/vicon/{topic_name}', PoseWithCovarianceStamped, partial(self._cb_obstacle_state, idx), queue_size=1)
                    rospy.wait_for_message(f'/vicon/{topic_name}', PoseWithCovarianceStamped, timeout=10)

                    additions.append(obstacle)

        return additions

    def run(self):
        for idx, obstacle_state in enumerate(self.obstacle_states):
            obs_state = torch.Tensor([*obstacle_state, 0., 0., 0., 0., 0., 0.])
            self.simulation.set_root_state_tensor_by_actor_idx(obs_state, idx + 1)

        self.simulation.reset_robot_state(self.robot_q, self.robot_q_dot)
        self.simulation.step()

    def destroy(self):
        self.simulation.stop_sim()

    def _cb_robot_state(self, msg):
        curr_pos = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y])
        curr_ori = msg.pose.pose.orientation

        _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

        self.robot_q = torch.tensor([curr_pos[0], curr_pos[1], curr_yaw])

        if self.robot_prev_msg is not None:
            prev_pos = torch.tensor([self.robot_prev_msg.pose.pose.position.x, self.robot_prev_msg.pose.pose.position.y])
            prev_ori = self.robot_prev_msg.pose.pose.orientation

            _, _, prev_yaw = euler_from_quaternion([prev_ori.x, prev_ori.y, prev_ori.z, prev_ori.w])

            delta_t = msg.header.stamp.to_sec() - self.robot_prev_msg.header.stamp.to_sec()

            linear_velocity = (curr_pos - prev_pos) / delta_t
            angular_velocity = (curr_yaw - prev_yaw) / delta_t

            cos_yaw = torch.cos(torch.tensor([curr_yaw]))
            sin_yaw = torch.sin(torch.tensor([curr_yaw]))
            self.robot_R = torch.tensor([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

            robot_xy_dot = torch.tensor([[linear_velocity[0]], [linear_velocity[1]]])
            robot_xy_dot = torch.matmul(torch.transpose(self.robot_R, 0, 1), robot_xy_dot)
            
            self.robot_q_dot = torch.tensor([*robot_xy_dot, angular_velocity])
            
        self.robot_prev_msg = msg

    def _cb_obstacle_state(self, idx, msg):
        pos, ori = msg.pose.pose.position, msg.pose.pose.orientation
        self.obstacle_states[idx] = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))
