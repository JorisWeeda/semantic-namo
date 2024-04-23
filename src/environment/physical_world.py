from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import ActorWrapper     # type: ignore
from control.mppi_isaac.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner       # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig          # type: ignore

from motion import Dingo
from scheduler import Objective

import random
import rospy
import roslib
import torch
import yaml

import numpy as np

from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped


class PhysicalWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, objectives, controller):        
        self.objectives = objectives
        self.controller = controller

        self.params = params
        self.config = config

        self.pos_tolerance = params['controller']['pos_tolerance']
        self.yaw_tolerance = params['controller']['yaw_tolerance']

        self.robot = Dingo()

        self.robot_prev_msg = None
        self.robot_q_dot = None
        self.robot_q = None
        self.robot_R = None

        self._goal = None
        self._mode = None

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

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}
        
        objective = Objective(config["mppi"].u_min, config["mppi"].u_max)

        controller = MPPIisaacPlanner(config, objective)
        return cls(params, config, objective, controller)

    def configure(self, robot_name):
        rospy.Subscriber(f'/vicon/{robot_name}', PoseWithCovarianceStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseWithCovarianceStamped, timeout=10)

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

                obstacle["init_ori"] = init_ori.tolist()
                obstacle["init_pos"] = init_pos.tolist()

                additions.append(obstacle)

        return additions

    def run(self):
        if self.robot_q is not None:
            action = self.controller.compute_action(self.robot_q, self.robot_q_dot)
            action = self._world_to_robot_frame(np.array(action), self.robot_R)

            lin_x, lin_y, ang_z = action[0], action[1], action[2]

            if not self.is_goal_reached:
                self.robot.move(lin_x, lin_y, ang_z)
            else:
                rospy.loginfo_throttle(1, "The goal is reached, no action applied to the robot.")


    def update_objective(self, goal, mode=(0, 0)):
        self._goal = goal
        self._mode = mode

        quaternions = self.yaw_to_quaternion(goal[2])
        tensor_goal = torch.tensor([goal[0], goal[1], 0., *quaternions],)
        tensor_mode = torch.tensor([mode[0], mode[1]])
        
        self.controller.update_objective_goal(tensor_goal)
        rospy.loginfo(f"Objective has new goal set : {tensor_goal}")
        rospy.loginfo(f"Objective expressed as state : {goal}")

        self.controller.update_objective_mode(tensor_mode)
        rospy.loginfo(f"Objective has new mode set : {mode}")

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
            
            self.check_goal_reached()
        else:
            self.update_objective((*curr_pos, curr_yaw))

        self.robot_prev_msg = msg

    def check_goal_reached(self):
        if self._goal is None:
            return None

        rob_yaw = self.robot_q[2]
        rob_pos = self.robot_q[:2]

        distance = np.linalg.norm(rob_pos - self._goal[:2])
        rotation = np.abs(rob_yaw - self._goal[2])

        self.is_goal_reached = False
        if distance < self.pos_tolerance and rotation < self.yaw_tolerance:
            self.is_goal_reached = True

    @staticmethod
    def _world_to_robot_frame(action, robot_R):
        action[:2] = robot_R.T.dot(action[:2].T)
        return action

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw)
