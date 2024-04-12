
import roslib
import yaml

import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from scheduler.global_planner import Planner


class Scheduler:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, robot_goal_pos, global_planner):
        self.global_planner = global_planner
        self.robot_goal_pos = robot_goal_pos

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
        
        global_planner = Planner(range_x, range_y, mass_threshold, path_inflation)
        return cls(robot_goal_pos, global_planner)
            
    def generate_path(self, sim):
        nodes, edges, polygons = self.global_planner.graph(sim, self.robot_goal_pos)