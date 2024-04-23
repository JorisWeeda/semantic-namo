from isaacgym import gymapi

import torch
import numpy as np
import networkx as nx

from shapely.geometry import Point, LineString, Polygon


class Planner:
    def __init__(self, range_x, range_y, mass_threshold, path_inflation, path_step_size):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation
        self.path_step_size = path_step_size
    
    def graph(self, sim, goal, margin=0.01):
        shapes, masses = self.generate_polygons(sim)
        stationary_polygons = [polygon for name, polygon in shapes.items() if masses[name] >= self.mass_threshold]

        nodes = self.generate_nodes(stationary_polygons)

        start = self.get_robot_pos(sim)
        nodes = np.vstack((start, nodes))
        nodes = np.vstack((nodes, goal))

        shapes, masses = self.generate_polygons(sim, self.path_inflation - margin)
        stationary_polygons = [polygon for name, polygon in shapes.items() if masses[name] >= self.mass_threshold]

        edges = self.generate_edges(nodes, stationary_polygons)

        graph = nx.Graph()
        for i, (x, y) in enumerate(nodes):
            graph.add_node(i, pos=(x, y))

        for edge in edges:
            graph.add_edge(edge[0], edge[1], length=edge[2])

        shortest_path = nx.shortest_path(graph, source=0, target=len(nodes)-1, weight='length')
        shortest_path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]

        barrier = self.filter_barriers(nodes, shortest_path_edges, shapes)
        shapes, masses = self.generate_polygons(sim, 0.)

        return shortest_path, nodes, edges, shapes, masses, barrier

    def generate_polygons(self, sim, inflation=None, threshold=None):
        inflation = self.path_inflation if inflation is None else inflation
        threshold = self.mass_threshold if threshold is None else threshold

        masses, shapes = {}, {}
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
            translate_corners = np.add(rotate_corners, obs_pos)
            
            mass = self.get_actor_mass(sim, actor)
            name = self.get_actor_name(sim, actor)
            
            shapes[name] = Polygon(translate_corners)
            masses[name] = mass

        return shapes, masses

    def generate_nodes(self, polygons):
        nodes = np.empty((0, 2), dtype='float')

        if polygons:
            corner_points = self.get_corner_points(polygons, self.range_x, self.range_y)
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

    @staticmethod
    def get_corner_points(polygons, x_limit, y_limit):
        corner_nodes = []
        for polygon in polygons:
            for corner in polygon.exterior.coords[:-1]:
                if x_limit[0] < corner[0] < x_limit[1]:
                    if y_limit[0] < corner[1] < y_limit[1]:
                        corner_nodes.append(corner)

        return np.array(corner_nodes)

    @staticmethod
    def get_intersection_points(polygons):
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

    @staticmethod
    def filter_nodes(nodes, polygons, radius=0.1):
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
    def filter_barriers(nodes, edges, shapes):
        blocking_shapes = {}
        for edge in edges:
            node_i, node_j = nodes[int(edge[0])], nodes[int(edge[1])]
            edge_line = LineString([node_i, node_j])

            for name, polygon in shapes.items():
                if edge_line.intersects(polygon):
                    blocking_shapes[name] = polygon

        barrier = []
        visited = []

        for name, polygon in blocking_shapes.items():
            if name in visited:
                continue

            new_barrier = {name: polygon}
            visited.append(name)

            for nested_name, nested_polygon in blocking_shapes.items():
                if name == nested_name:
                    continue

                if polygon.intersects(nested_polygon):
                    new_barrier[nested_name] = nested_polygon
                    visited.append(name)
            
            barrier.append(new_barrier)
        return barrier

    @staticmethod
    def get_actor_pos(sim, actor):
        rb_state = sim._gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return np.array([rb_state["pose"]["p"]["x"], rb_state["pose"]["p"]["y"]], dtype=np.float32)

    @staticmethod
    def get_actor_yaw(sim, actor):
        rb_state = sim._gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return gymapi.Quat.to_euler_zyx(rb_state["pose"]["r"])[-1]

    @staticmethod
    def get_actor_mass(sim, actor):
        rigid_body_property = sim._gym.get_actor_rigid_body_properties(sim.envs[0], actor)[0]
        return float('inf') if sim.env_cfg[actor].fixed else rigid_body_property.mass

    @staticmethod
    def get_actor_name(sim, actor):
        return sim._gym.get_actor_name(sim.envs[0], actor)
    
    @staticmethod
    def get_robot_pos(sim):
        rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        return rob_pos[0].numpy()
