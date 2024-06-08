from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import ActorWrapper     # type: ignore
from control.mppi_isaac.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner       # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig          # type: ignore

from motion import Dingo
from scheduler import Objective

import io
import rospy
import roslib
import torch
import yaml

from functools import partial
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped


class PhysicalWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    MSG_TIMEOUT = 3
    
    def __init__(self, params, config, controller):        
        self.controller = controller
        self.env_config = None

        self.params = params
        self.config = config

        self.pos_tolerance = params['controller']['pos_tolerance']
        self.yaw_tolerance = params['controller']['yaw_tolerance']

        self.robot = Dingo()

        self.obstacle_states = []

        self.robot_prev_msg = None
        self.robot_q_dot = torch.zeros(3)
        self.robot_R = torch.zeros(2, 2)
        self.robot_q = torch.zeros(3)

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
        
        objective = Objective(config["mppi"].u_min, config["mppi"].u_max, config["mppi"].device)

        controller = MPPIisaacPlanner(config, objective)
        return cls(params, config, controller)

    def configure(self, robot_name):
        additions = self.create_additions()
        
        self.controller.add_to_env(additions)
        self.env_config = additions
        
        rospy.Subscriber(f'/vicon/{robot_name}', PoseWithCovarianceStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseWithCovarianceStamped, timeout=10)

        self.update_objective(self.robot_q.tolist(), (0., 0.))

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
        if self.robot_prev_msg is not None:
            action = self.controller.compute_action(self.robot_q, self.robot_q_dot)
            action[:2] = torch.matmul(self.robot_R.T, action[:2])

            if not self.is_goal_reached:
                self.robot.move(*action)
            else:
                rospy.loginfo_throttle(1, "The goal is reached, no action applied to the robot.")

    def update_objective(self, goal, mode=[0, 0]):
        self._goal = torch.tensor(goal)
        self._mode = torch.tensor(mode)

        quaternions = self.yaw_to_quaternion(goal[2])
        q, q_dot = self.robot_q, self.robot_q_dot

        tensor_init = torch.tensor([q[0], q_dot[0], q[1], q_dot[1], q[2], q_dot[2]])
        tensor_goal = torch.tensor([goal[0], goal[1], 0., *quaternions])
        tensor_mode = torch.tensor([mode[0], mode[1]])

        rospy.loginfo(f"New starting state: {tensor_init}")
        rospy.loginfo(f"New objective goal: {tensor_goal}")
        rospy.loginfo(f"New objective mode: {tensor_mode}")

        self.controller.update_objective(tensor_init, tensor_goal, tensor_mode)

    def check_goal_reached(self):
        if self._goal is None:
            return None

        self.is_goal_reached = False
        if torch.linalg.norm(self._goal[:2] - self.robot_q[:2]) < self.pos_tolerance :
            self.is_goal_reached = True

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
            
            self.robot_q_dot = torch.tensor([linear_velocity[0], linear_velocity[1], angular_velocity])
            
            self.check_goal_reached()

        self.robot_prev_msg = msg

    def _cb_obstacle_state(self, idx, msg):
        pos, ori = msg.pose.pose.position, msg.pose.pose.orientation
        self.obstacle_states[idx] = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

    def get_robot_dofs(self):
        q, q_dot = self.robot_q, self.robot_q_dot
        return torch.Tensor([q[0], q_dot[0], q[1], q_dot[1], q[2], q_dot[2]])

    def get_actor_states(self):
        return self.env_config[1:], torch.Tensor(self.obstacle_states)
    
    def get_rollout_states(self):
        return self.bytes_to_torch(self.controller.get_states())

    def get_rollout_best_state(self):
        return self.bytes_to_torch(self.controller.get_n_best_samples())

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw)

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
    