from isaacgym import gymapi

import torch
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon as MatPlotPolygon
from shapely.geometry import Point, LineString, Polygon


class Planner:
    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation
    
    def graph(self, sim, goal, margin=0.01):
        adjustable_polygons, stationary_polygons = self.generate_polygons(sim)
        
        if stationary_polygons:
            nodes = self.generate_nodes(stationary_polygons)
        
        start = self.get_robot_pos(sim)
        nodes = np.vstack((start, nodes))
        nodes = np.vstack((nodes, goal))
        
        inflation = self.path_inflation - margin 

        _, stationary_polygons = self.generate_polygons(sim, inflation)
        edges = self.generate_edges(nodes, stationary_polygons)

        # ------------------------------------
        graph = nx.Graph()
        for i, (x, y) in enumerate(nodes):
            graph.add_node(i, pos=(x, y))

        for edge in edges:
            graph.add_edge(edge[0], edge[1], length=edge[2])

        shortest_path = nx.shortest_path(graph, source=0, target=len(nodes)-1, weight='length')
        shortest_path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]


        blocking_obstacles = []
        for edge in shortest_path_edges:
            node_i, node_j = nodes[int(edge[0])], nodes[int(edge[1])]
            edge_line = LineString([node_i, node_j])
            for polygon in adjustable_polygons:
                if edge_line.intersects(polygon):
                    blocking_obstacles.append(polygon)
  
        _, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

        ax.scatter(nodes[:, 0], nodes[:, 1], color='green')

        for polygon in stationary_polygons:
            patch = MatPlotPolygon([(point[0], point[1]) for point in polygon.exterior.coords], color='black', lw=1, alpha=0.5)
            ax.add_patch(patch)

        for polygon in blocking_obstacles:
            patch = MatPlotPolygon([(point[0], point[1]) for point in polygon.exterior.coords], color='orange', lw=1, alpha=0.5)
            ax.add_patch(patch)

        for edge in edges:
            node_i, node_j = nodes[int(edge[0])], nodes[int(edge[1])]
            ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], color='blue', linewidth=0.1)

        for edge in shortest_path_edges:
            node_i, node_j = nodes[int(edge[0])], nodes[int(edge[1])]
            ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], color='green', linewidth=3)

        ax.autoscale_view()
        ax.set_title("Environment with Obstacles and Robot")
        ax.grid(True)

        plt.show()

        # ------------------------------------

        return nodes, edges, stationary_polygons

    def generate_polygons(self, sim, inflation=None, threshold=None):
        inflation = self.path_inflation if inflation is None else inflation
        threshold = self.mass_threshold if threshold is None else threshold

        lower_polygons, higher_polygons = [], []
        for actor in range(1, len(sim.env_cfg)):
            actor_wrapper = sim.env_cfg[actor]

            obs_pos = self.get_actor_pos(sim, actor)
            obs_rot = self.get_actor_yaw(sim, actor)

            inflated_size_x = actor_wrapper.size[0] + 2 * inflation
            inflated_size_y = actor_wrapper.size[1] + 2 * inflation

            corners = np.array([[-inflated_size_x / 2, -inflated_size_y / 2],
                                [inflated_size_x / 2, -inflated_size_y / 2],
                                [inflated_size_x / 2, inflated_size_y / 2],
                                [-inflated_size_x / 2, inflated_size_y / 2],
                                [-inflated_size_x / 2, -inflated_size_y / 2]])

            rotation_matrix = np.array([[np.cos(obs_rot), -np.sin(obs_rot)],
                                        [np.sin(obs_rot), np.cos(obs_rot)]])

            rotate_corners = np.dot(corners, rotation_matrix)
            translate_corners = rotate_corners + obs_pos

            if self.get_actor_mass(sim, actor) < threshold:
                lower_polygons.append(Polygon(translate_corners))
            else:
                higher_polygons.append(Polygon(translate_corners))

        return lower_polygons, higher_polygons

    def generate_nodes(self, polygons):
        nodes = np.empty((0, 2), dtype='float')

        corner_points = self.get_corner_points(polygons)
        nodes = np.vstack((nodes, corner_points))

        intersect_points = self.get_intersection_points(polygons)
        nodes = np.vstack((nodes, intersect_points))

        nodes = self.filter_nodes(nodes, polygons)
        return nodes
    
    def generate_edges(self, nodes, polygons):
        edges = np.empty((0, 3))

        for i in range(nodes.shape[0]):
            for j in range(i + 1, nodes.shape[0]):
                if i != j:
                    edge_line = LineString([nodes[i, :], nodes[j, :]])
                    if not any(edge_line.intersects(polygon) for polygon in polygons):
                        edges = np.vstack((edges, [int(i), int(j), edge_line.length]))
        return edges

    def get_corner_points(self, polygons):
        corner_nodes = []
        for polygon in polygons:
            for corner in polygon.exterior.coords[:-1]:
                if self.range_x[0] < corner[0] < self.range_x[1]:
                    if self.range_y[0] < corner[1] < self.range_y[1]:
                        corner_nodes.append(corner)

        return np.array(corner_nodes)

    def get_intersection_points(self, polygons):
        intersection_points = []
        for i, polygon_i in enumerate(polygons):
            for j, polygon_j in enumerate(polygons):
                if i != j:
                    intersection = polygon_i.intersection(polygon_j)
                    if isinstance(intersection, Point):
                        intersection_points.append([intersection.x, intersection.y])
                    elif isinstance(intersection, LineString):
                        intersection_points.extend([[point[0], point[1]] for point in intersection.coords])
                    elif isinstance(intersection, Polygon):
                        intersection_points.extend([[point[0], point[1]] for point in intersection.exterior.coords])
        return np.array(intersection_points)

    def filter_nodes(self, nodes, polygons, radius=0.1):
        filtered_nodes = []

        for node in nodes:
            point = Point(node)

            is_within_polygon = any(polygon.contains(point) for polygon in polygons)
            if not is_within_polygon:
                filtered_nodes.append(node)
            else:
                is_valid = any(not any(polygon.contains(Point(node[0] + dx, node[1] + dy)) for polygon in polygons)
                               for dx in np.arange(-radius, radius + 0.001, 0.01)
                               for dy in np.arange(-radius, radius + 0.001, 0.01))
                if is_valid:
                    filtered_nodes.append(node)

        return np.array(filtered_nodes)

    @staticmethod
    def get_actor_pos(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return np.array([rb_state["pose"]["p"]["x"], rb_state["pose"]["p"]["y"]], dtype=np.float32)

    @staticmethod
    def get_actor_yaw(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return gymapi.Quat.to_euler_zyx(rb_state["pose"]["r"])[-1]

    @staticmethod
    def get_actor_mass(sim, actor):
        if sim.env_cfg[actor].fixed:
            return float('inf')

        rigid_body_property = sim.gym.get_actor_rigid_body_properties(sim.envs[0], actor)[0]
        return rigid_body_property.mass

    @staticmethod
    def get_robot_pos(sim):
        rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        return rob_pos[0].numpy()
