

import rospy
import roslib
import yaml

import numpy as np

from scheduler.global_planner import Planner


class Scheduler:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, robot_goal_pos, global_planner):
        self.global_planner = global_planner
        self.robot_goal_pos = robot_goal_pos

        self.waypoints = None
        self.point_idx = 0

    @classmethod
    def create_scheduler(cls, layout):
        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}

        range_x = params['range_x']
        range_y = params['range_y']

        robot_goal_pos = params['scheduler']['goal']

        mass_threshold = params['scheduler']['mass_threshold']
        path_inflation = params['scheduler']['path_inflation']
        path_step_size = params['scheduler']['path_step_size']
        
        global_planner = Planner(range_x, range_y, mass_threshold, path_inflation, path_step_size)
        return cls(robot_goal_pos, global_planner)

    def generate_tasks(self, sim):
        shortest_path, nodes, _, _, _, _ = self.global_planner.graph(sim, self.robot_goal_pos)
        self.waypoints = np.array([nodes[int(node_idx)] for node_idx in shortest_path])
        rospy.loginfo(f"New set of waypoints calculated: {self.waypoints}")

    def get_next_waypoint(self):
        if self.waypoints is None:
            return None
        
        self.point_idx += 1
        if self.point_idx == len(self.waypoints):
            return None
        
        return self.waypoints[self.point_idx, :]

    def is_finished(self):
        if self.waypoints is not None and self.point_idx == len(self.waypoints):
            return True
        
        return False
