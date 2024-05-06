

import roslib
import yaml

import networkx as nx
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

        robot_goal_pos = params['goal']

        mass_threshold = params['scheduler']['mass_threshold']
        path_inflation = params['scheduler']['path_inflation']

        global_planner = Planner(range_x, range_y, mass_threshold, path_inflation)
        return cls(robot_goal_pos, global_planner)

    def generate_tasks(self, sim):
        graph, _, _ = self.global_planner.graph(sim, self.robot_goal_pos)

        init_node = self.find_closest_node(graph, self.global_planner.get_robot_pos(sim))
        goal_node = self.find_closest_node(graph, self.robot_goal_pos)

        shortest_path = nx.shortest_path(graph, source=init_node, target=goal_node, weight=lambda _, waypoint_node, edge_data: 
                                         self.custom_weight_function(waypoint_node, edge_data, graph))

        nodes = np.array([(*data['pos'], data['cost'])  for _, data in graph.nodes(data=True)])
        edges = np.array([(graph.nodes[u]['pos'], graph.nodes[v]['pos']) for u, v, _ in graph.edges(data=True)])

        self.waypoints = np.array([nodes[int(i)] for i in shortest_path])

        return shortest_path, nodes, edges
    
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

    @staticmethod
    def find_closest_node(graph, coordinate):
        closest_node = None
        min_distance = float('inf')

        for node, pos in graph.nodes(data='pos'):
            distance = np.linalg.norm(np.array(pos) - np.array(coordinate))
            
            if distance < min_distance:
                closest_node = node
                min_distance = distance

        return closest_node
    
    @staticmethod
    def custom_weight_function(waypoint_id, edge_data, graph):
        waypoint_cost = graph.nodes[waypoint_id]['cost']
        length_to_waypoint = edge_data['length']
        
        combined_weight = length_to_waypoint + waypoint_cost
        return combined_weight
