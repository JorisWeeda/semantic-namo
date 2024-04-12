from control.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import rospy
import roslib
import yaml

import numpy as np

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped

from isaacgym import gymapi


class Viewer:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation):
        self.simulation = simulation

        self.params = params
        self.config = config

        self.robot_q = None

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, robot_name: str):
        viewer = cls.create(config, layout)
        viewer.configure(robot_name)
        return viewer

    @classmethod
    def create(cls, config, layout):
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

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}

        return cls(params, config, simulation)
    
    def configure(self, robot_name):
        rospy.Subscriber(f'/vicon/{robot_name}', PoseWithCovarianceStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseWithCovarianceStamped, timeout=10)

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation.gym, self.simulation.viewer, cam_pos, cam_tar)

    def run(self):
        self.simulation.reset_robot_state(self.robot_q, np.zeros_like(self.robot_q))
        self.simulation.step()

    def destroy(self):
        self.simulation.gym.destroy_viewer(self.simulation.viewer)
        self.simulation.gym.destroy_sim(self.simulation.sim)

    def _cb_robot_state(self, msg):
        curr_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        curr_ori = msg.pose.pose.orientation

        _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

        self.robot_q = np.array([curr_pos[0], curr_pos[1], curr_yaw])

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))
