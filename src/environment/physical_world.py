from control.mppiisaac.planner.isaacgym_wrapper import ActorWrapper     # type: ignore
from control.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner       # type: ignore

import hydra
import rospy
import roslib
import yaml

import numpy as np

from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

from motion import Dingo
from scheduler import Objective


class PhysicalWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, objectives, controller):        
        self.objectives = objectives
        self.controller = controller

        self.params = params
        self.config = config

        self.robot = Dingo()

        self.robot_R = None
        self.robot_q = None
        self.robot_q_dot = None

    @hydra.main(version_base=None, config_path="config")
    def build(config):
        world = PhysicalWorld.create(config, config["world"])
        world.configure()
        return world

    @classmethod
    def create(cls, config, world):
        actors=[]
        for actor_name in config["actors"]:
            with open(f'{cls.PKG_PATH}/config/actors/{actor_name}.yaml') as f:
                actors.append(ActorWrapper(**yaml.load(f, Loader=yaml.SafeLoader)))

        world_config = f'{cls.PKG_PATH}/config/worlds/{world}.yaml'
        with open(world_config, "r") as stream:
            params = yaml.safe_load(stream)

        objectives = Objective()
        controller = MPPIisaacPlanner(config, objectives)

        return cls(params, config, objectives, controller)

    def configure(self):
        rospy.Subscriber("/optitrack_state_estimator/Dingo/state", Odometry, self._robot_state_cb, queue_size=1,)
        rospy.wait_for_message('/optitrack_state_estimator/Dingo/state', Odometry, timeout=10)

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
        action = self.controller.compute_action(self.robot_q, self.robot_q_dot)
        action = self._world_to_robot_frame(np.array(action), self.robot_R)

        lin_x, lin_y, ang_z = action[0], action[1], action[2]
        self.robot.move(lin_x, lin_y, ang_z)

    def _robot_state_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular

        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        self.robot_R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.robot_q = np.array([pos.x, pos.y, yaw])
        self.robot_q_dot = np.array([lin.x, lin.y, ang.z,])

    @staticmethod
    def _world_to_robot_frame(action, robot_R):
        action[:2] = robot_R.T.dot(action[:2].T)
        return action
