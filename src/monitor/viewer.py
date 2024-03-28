from control.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import rospy
import roslib
import yaml

import numpy as np

from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

from isaacgym import gymapi


class Viewer:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation):
        self.simulation = simulation

        self.params = params
        self.config = config

        self.robot_q = None

    @classmethod
    def build(cls, config: ExampleConfig, layout: str):
        viewer = cls.create(config, layout)
        viewer.configure()
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

        world_config = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config, "r") as stream:
            params = yaml.safe_load(stream)

        return cls(params, config, simulation)
    
    def configure(self):
        rospy.Subscriber("/optitrack_state_estimator/Dingo/state", Odometry, self._robot_state_cb, queue_size=1,)
        rospy.wait_for_message('/optitrack_state_estimator/Dingo/state', Odometry, timeout=10)

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation.gym, self.simulation.viewer, cam_pos, cam_tar)

    def run(self):
        self.simulation.reset_robot_state(self.robot_q, np.zeros_like(self.robot_q))
        self.simulation.step()

    def destroy(self):
        self.simulation.gym.destroy_viewer(self.simulation.viewer)
        self.simulation.gym.destroy_sim(self.simulation.sim)

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))

    def _robot_state_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular

        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        self.robot_R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.robot_q = np.array([pos.x, pos.y, yaw])
        self.robot_q_dot = np.array([lin.x, lin.y, ang.z,])
