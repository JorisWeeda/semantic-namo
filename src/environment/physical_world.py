from control.mppiisaac.planner.isaacgym_wrapper import ActorWrapper     # type: ignore
from control.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner       # type: ignore
from control.mppiisaac.utils.config_store import ExampleConfig          # type: ignore

from motion import Dingo
from scheduler import Objective

import rospy
import roslib
import torch
import yaml

import numpy as np

from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped


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
        self.robot_prev_msg = None

        self.is_goal_reached = False

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, robot_name: str):
        world = PhysicalWorld.create(config, layout)
        world.configure(robot_name)
        return world
    
    @classmethod
    def create(cls, config, layout):
        actors=[]
        for actor_name in config["actors"]:
            with open(f'{cls.PKG_PATH}/config/actors/{actor_name}.yaml') as f:
                actors.append(ActorWrapper(**yaml.load(f, Loader=yaml.SafeLoader)))

        world_config = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config, "r") as stream:
            params = yaml.safe_load(stream)

        objectives = Objective()
        controller = MPPIisaacPlanner(config, objectives)

        return cls(params, config, objectives, controller)

    def configure(self, robot_name):
        rospy.Subscriber(f'/vicon/{robot_name}', PoseWithCovarianceStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseWithCovarianceStamped, timeout=10)

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
        if self.robot_q is not None:
            action = self.controller.compute_action(self.robot_q, self.robot_q_dot)
            action = self._world_to_robot_frame(np.array(action), self.robot_R)

            lin_x, lin_y, ang_z = action[0], action[1], action[2]

            if not self.is_goal_reached:
                self.robot.move(lin_x, lin_y, ang_z)

    def _cb_robot_state(self, msg):
        curr_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        curr_ori = msg.pose.pose.orientation

        _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

        if self.robot_prev_msg is not None:
            prev_pos = np.array([self.robot_prev_msg.pose.pose.position.x, self.robot_prev_msg.pose.pose.position.y])
            prev_ori = self.robot_prev_msg.pose.pose.orientation

            _, _, prev_yaw = euler_from_quaternion([prev_ori.x, prev_ori.y, prev_ori.z, prev_ori.w])

            delta_t = msg.header.stamp.to_sec() - self.robot_prev_msg.header.stamp.to_sec()

            linear_velocity = (curr_pos - prev_pos) / delta_t
            angular_velocity = (curr_yaw - prev_yaw) / delta_t

            self.robot_R = np.array([[np.cos(curr_yaw), -np.sin(curr_yaw)], [np.sin(curr_yaw), np.cos(curr_yaw)]])
            self.robot_q = np.array([curr_pos[0], curr_pos[1], curr_yaw])
            self.robot_q_dot = np.array([linear_velocity[0], linear_velocity[1], angular_velocity])

            error_x, error_y, _ = np.abs(self.robot_q - self.objectives.goal_state)
            if error_x < 0.05 and error_y < 0.05:
                rospy.loginfo_once(f"Goal succeeded, Ex: {error_x} and Ey: {error_y}")
                self.is_goal_reached = True 

        else:
            goal = torch.tensor([*curr_pos, 0., curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])
            self.controller.update_objective_goal(goal)

        self.robot_prev_msg = msg

    @staticmethod
    def _world_to_robot_frame(action, robot_R):
        action[:2] = robot_R.T.dot(action[:2].T)
        return action
