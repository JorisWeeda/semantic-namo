from isaacgym import gymapi

import rospy
import torch
import numpy as np
import networkx as nx

from itertools import combinations

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points


class Planner:
    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation
    
    def graph(self, sim, goal, margin=0.05):
        inflated_shapes, masses = self.generate_polygons(sim)
        nodes = self.generate_nodes(inflated_shapes.values())

        init_node = (*self.get_robot_pos(sim), 0.)
        goal_node = (*goal, 0.)

        nodes = np.vstack((init_node, nodes))
        nodes = np.vstack((nodes, goal_node))

        reduced_inflated_shapes, masses = self.generate_polygons(sim, self.path_inflation - margin)
        edges = self.generate_edges(nodes, reduced_inflated_shapes.values())

        graph = nx.Graph()

        for i, (x, y, cost) in enumerate(nodes):
            graph.add_node(i, pos=(x, y))
            graph.nodes[i]['cost'] = cost

        for edge in edges:
            source, target, length = edge
            graph.add_edge(source, target, length=length)

        rospy.loginfo(f'Is the occupation snapshot fully connected: {nx.is_connected(graph)}')

        non_inflated_shapes, masses = self.generate_polygons(sim, 0.)
        
        stationary_polygons = {name: polygon for name, polygon in inflated_shapes.items() if masses[name] >= self.mass_threshold}
        adjustable_polygons = {name: polygon for name, polygon in non_inflated_shapes.items() if masses[name] < self.mass_threshold}

        passages = self.generate_passages({**stationary_polygons, **adjustable_polygons}, masses)
        graph = self.add_passages_to_graph(graph, passages)

        return graph, non_inflated_shapes, masses

    def generate_polygons(self, sim, overwrite_inflation=None):
        overwrite_inflation = self.path_inflation if overwrite_inflation is None else overwrite_inflation

        masses, shapes = {}, {}
        for actor in range(1, len(sim.env_cfg)):
            actor_wrapper = sim.env_cfg[actor]

            obs_pos = self.get_actor_pos(sim, actor)
            obs_rot = self.get_actor_yaw(sim, actor)

            inflated_size_x = actor_wrapper.size[0] + 2 * overwrite_inflation
            inflated_size_y = actor_wrapper.size[1] + 2 * overwrite_inflation

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
        nodes = np.empty((0, 3), dtype='float')

        if polygons:
            corner_points = self.get_corner_points(polygons, self.range_x, self.range_y)
            if len(corner_points) != 0:
                corner_points = np.hstack((corner_points, np.zeros((corner_points.shape[0], 1))))
                nodes = np.vstack((nodes, corner_points))

            intersect_points = self.get_intersection_points(polygons, self.range_x, self.range_y)
            if len(intersect_points) != 0:
                intersect_points = np.hstack((intersect_points, np.zeros((intersect_points.shape[0], 1))))
                nodes = np.vstack((nodes, intersect_points))

            nodes = self.filter_nodes(nodes, polygons)

        return nodes

    def generate_passages(self, shapes, masses):
        passages = np.empty((0, 4), dtype='float')

        id_pairs = self.find_close_pairs(shapes, self.path_inflation)
        for id_1, id_2 in id_pairs:
            if masses[id_1] >= self.mass_threshold and masses[id_2] >= self.mass_threshold:
                continue

            heavy_id, light_id = [id_1, id_2] if masses[id_1] >= masses[id_2] else [id_2, id_1]
            heavy_ob, light_ob = shapes[heavy_id], shapes[light_id]

            point_heavy_ob, point_light_ob = nearest_points(heavy_ob.boundary, light_ob.boundary)

            passage_center_x = (point_heavy_ob.x + point_light_ob.centroid.x) / 2
            passage_center_y = (point_heavy_ob.y + point_light_ob.centroid.y) / 2
            passage_centroid = Point(passage_center_x, passage_center_y) 

            point_centroid_light_ob, point_closest_heavy_ob = nearest_points(light_ob.centroid, heavy_ob.boundary)

            centroid_line = LineString([point_centroid_light_ob, point_closest_heavy_ob])
            nearest_point = centroid_line.interpolate(centroid_line.project(passage_centroid))

            search_distance = self.max_distance_polygon(light_ob)
            passages = np.vstack((passages, (nearest_point.x, nearest_point.y, masses[light_id], search_distance)))
        return passages

    def generate_edges(self, nodes, polygons):
        edges = []

        node_combinations = combinations(range(nodes.shape[0]), 2)
        for i, j in node_combinations:
            edge_line = LineString([nodes[i], nodes[j]])

            if not any(edge_line.intersects(polygon) for polygon in polygons):
                edges.append([int(i), int(j), edge_line.length])

        return np.array(edges)

    @staticmethod
    def get_corner_points(polygons, x_limit, y_limit):
        corner_nodes = []
        for polygon in polygons:
            for corner in polygon.exterior.coords[:-1]:
                if x_limit[0] <= corner[0] <= x_limit[1]:
                    if y_limit[0] <= corner[1] <= y_limit[1]:
                        corner_nodes.append(corner)

        return np.array(corner_nodes)

    @staticmethod
    def get_intersection_points(polygons, x_limit, y_limit):
        intersection_points = []
        for i, polygon_i in enumerate(polygons):
            for j, polygon_j in enumerate(polygons):
                if i != j:
                    intersection = polygon_i.intersection(polygon_j)

                    if isinstance(intersection, Point):
                        if x_limit[0] <= intersection.x <= x_limit[1] and y_limit[0] <= intersection.y <= y_limit[1]:
                            intersection_points.append([intersection.x, intersection.y])

                    elif isinstance(intersection, LineString):
                        for point in intersection.coords:
                            if x_limit[0] <= point[0] <= x_limit[1] and y_limit[0] <= point[1] <= y_limit[1]:
                                intersection_points.append([point[0], point[1]])

                    elif isinstance(intersection, Polygon):
                        for point in intersection.exterior.coords:
                            if x_limit[0] <= point[0] <= x_limit[1] and y_limit[0] <= point[1] <= y_limit[1]:
                                intersection_points.append([point[0], point[1]])

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

        filtered_nodes = np.array(filtered_nodes)
        return np.unique(filtered_nodes, axis=0)

    @staticmethod
    def find_close_pairs(shapes, threshold):
        passage_pairs = []

        obstacles_id_pairs = list(combinations(shapes.keys(), 2))
        for id_1, id_2 in obstacles_id_pairs:
            obstacle_1, obstacle_2 = shapes[id_1], shapes[id_2]

            if obstacle_1.boundary.distance(obstacle_2.boundary) < threshold:
                passage_pairs.append((id_1, id_2))

        return np.array(passage_pairs)

    @staticmethod
    def add_passages_to_graph(graph, passages, shapes=None):
        num_existing_nodes = graph.number_of_nodes()

        for i, (x, y, cost, search_distance) in enumerate(passages):
            new_node_index = num_existing_nodes + i
            graph.add_node(new_node_index, pos=(x, y), cost=cost)

            new_node_pos = (x, y)
            for node, node_pos in graph.nodes(data='pos'):
                edge_line = LineString([node_pos, new_node_pos])
                if node != new_node_index and edge_line.length <= search_distance:
                    if not shapes: 
                        graph.add_edge(new_node_index, node, length=edge_line.length)
                    if shapes and not any(edge_line.intersects(polygon) for polygon in shapes.values()):
                        graph.add_edge(new_node_index, node, length=edge_line.length)

        return graph

    @staticmethod    
    def max_distance_polygon(polygon):
        vertices = list(polygon.exterior.coords)
        distance = [LineString([v_1, v_2]).length for v_1, v_2 in combinations(vertices, 2)]
        return max(distance)

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
