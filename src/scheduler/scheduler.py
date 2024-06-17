import rospy
import roslib
import yaml

import networkx as nx
import numpy as np

from scipy.interpolate import splprep, splev


from scheduler.global_planner import NVG, SVG, PRM, RRT


class Scheduler:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")
    
    def __init__(self, robot_goal_pos, spline_interval, spline_step, nvg_planner, svg_planner, prm_planner, rrt_planner):
        self.robot_goal_pos = robot_goal_pos
        
        self.spline_interval = spline_interval
        self.spline_step = spline_step

        self.nvg_planner = nvg_planner
        self.svg_planner = svg_planner
        self.prm_planner = prm_planner
        self.rrt_planner = rrt_planner

        self.waypoints = None
        self.new_index = None

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
        
        spline_interval = params['scheduler']['spline_interval']
        spline_step = params['scheduler']['spline_step']

        mass_threshold = params['scheduler']['mass_threshold']
        path_inflation = params['scheduler']['path_inflation']
        
        nvg_planner = NVG(range_x, range_y, mass_threshold, path_inflation)
        svg_planner = SVG(range_x, range_y, mass_threshold, path_inflation)
        prm_planner = PRM(range_x, range_y, mass_threshold, path_inflation)
        rrt_planner = RRT(range_x, range_y, mass_threshold, path_inflation)

        return cls(robot_goal_pos, spline_interval, spline_step, nvg_planner, svg_planner, prm_planner, rrt_planner)

    def generate_path(self, robot_dof, actors, mode='svg'):
        q_init = [robot_dof[0], robot_dof[2]]
        q_goal = self.robot_goal_pos

        if mode == 'nvg':
            rospy.loginfo(f"Planning from {q_init} to {q_goal} using nvg.")
            graph = self.nvg_planner.graph(q_init, q_goal, actors)
        elif mode == 'svg':
            rospy.loginfo(f"Planning from {q_init} to {q_goal} using svg.")
            graph = self.svg_planner.graph(q_init, q_goal, actors)
        elif mode == 'prm':
            rospy.loginfo(f"Planning from {q_init} to {q_goal} using PRM.")
            graph = self.prm_planner.graph(q_init, q_goal, actors)
        elif mode == 'rrt':
            rospy.loginfo(f"Planning from {q_init} to {q_goal} using RRT.")
            graph = self.rrt_planner.graph(q_init, q_goal, actors)
        else:
            raise TypeError(f'Mode {mode} not recognized as a global planner.')

        if not graph or not any(np.linalg.norm(np.array(node[1]) - np.array(q_goal)) < 1e-6 for node in graph.nodes(data='pos')):
            rospy.loginfo(f"Failed to create graph using {mode}.")
            return False, graph, np.empty((0, 3), dtype='float'), float(0)

        init_node = self.find_closest_node(graph, q_init)
        goal_node = self.find_closest_node(graph, q_goal)

        try:
            shortest_path = nx.astar_path(graph, source=init_node, target=goal_node,weight=lambda _, path_node, 
                                          edge_data: self.custom_weight_function(path_node, edge_data, graph))
        except nx.exception.NetworkXNoPath:
            rospy.loginfo(f"Failed to create shortest path using {mode}.")
            return False, graph, np.empty((0, 3), dtype='float'), float(0)

        waypoints = np.array([graph.nodes[path_node]['pos'] for path_node in shortest_path])

        self.waypoints = self.apply_parametric_spline(waypoints, self.spline_interval)
        cost = sum(graph[u][v]['length'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))

        return True, graph, self.waypoints, cost

    def get_next_waypoints(self, robot_dof):
        q_rob = np.array([robot_dof[0], robot_dof[2]])
        if self.waypoints is None: 
            return None
        
        distances = np.linalg.norm(self.waypoints - q_rob, axis=1)
        if len(distances) == 0:
            return None

        waypoint_index = min(np.argmin(distances) + 1, len(self.waypoints) - 1)
        horizon_index = min(waypoint_index + self.spline_step, len(self.waypoints))

        self.new_index = waypoint_index 
        return self.waypoints[waypoint_index: horizon_index, :]

    def is_finished(self):
        if  self.new_index is not None and  self.new_index >= len(self.waypoints) -1:
            self.waypoints, self.next_waypoints = None, None
            return True

        return False

    @staticmethod
    def apply_parametric_spline(waypoints, interval, order=1):
        length = Scheduler.calculate_total_length(waypoints)
        n = int(length / interval)

        x, y = waypoints[:, 0], waypoints[:, 1]
        m = len(x)

        k = min(order, m - 1)
        if k < 1:
            raise ValueError("Not enough waypoints to form a spline.")

        tck, _ = splprep([x, y], s=0, k=k, per=False)
        u_new = np.linspace(0, 1, n)

        interpolated_points = np.array(splev(u_new, tck))
        return interpolated_points.T

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
    def calculate_total_length(waypoints):
        diffs = np.diff(waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        total_length = np.sum(distances)
        return total_length

    @staticmethod
    def custom_weight_function(path_node, edge_data, graph):
        node_cost = graph.nodes[path_node]['cost']
        edge_cost = edge_data['length']
        return edge_cost + node_cost

