from isaacgym import gymapi
from control.mppi_isaac.mppiisaac.utils.conversions import quaternion_to_yaw

import rospy
import torch
import numpy as np
import networkx as nx

from itertools import combinations

from shapely import buffer, prepare
from shapely.affinity import rotate
from shapely.geometry import Point, MultiPoint, LineString, Polygon
from shapely.ops import nearest_points


class SRM:

    NEAREST_NEIGHBOURS = 4

    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation
    
    def graph(self, q_init, q_goal, actors):
        if len(actors[0]) > 1 and self.is_goal_blocked(q_init, q_goal, actors):
            rospy.loginfo("Goal is blocked by static obstacles.")
            return None

        actor_polygons, _ = self.generate_polygons(actors)
        avoid_obstacle, _ = self.generate_polygons(actors, self.path_inflation - 0.05)

        graph = nx.Graph()
        self.add_node_to_graph(graph, (*q_init, 0.), avoid_obstacle.values())

        actor_nodes = self.generate_nodes(actor_polygons.values())
        for node in actor_nodes:
            self.add_node_to_graph(graph, node, avoid_obstacle.values())

        self.add_node_to_graph(graph, (*q_goal, 0.), avoid_obstacle.values())

        try:
            nx.shortest_path(graph, source=0, target=len(graph.nodes) -1)
        except nx.NetworkXNoPath:
            rospy.loginfo("Avoidance is not possible, creating passage nodes.")
            pass

        inflated_shapes, masses = self.generate_polygons(actors)
        non_inflated_shapes, _ = self.generate_polygons(actors, 0.)

        stationary_polygons = {name: polygon for name, polygon in inflated_shapes.items() if masses[name] >= self.mass_threshold}
        adjustable_polygons = {name: polygon for name, polygon in non_inflated_shapes.items() if masses[name] < self.mass_threshold}

        passages = self.generate_passages({**stationary_polygons, **adjustable_polygons}, masses)
        graph = self.add_passages_to_graph(graph, passages)

        return graph

    def is_goal_blocked(self, q_init, q_goal, actors):
        stationary_actors_wrapper = [actor for actor in actors[0] if actor.mass >= self.mass_threshold]
        stationary_actors_state = torch.stack([state for actor, state in zip(actors[0], actors[1]) if actor.mass >= self.mass_threshold])

        stationary_actors = (stationary_actors_wrapper, stationary_actors_state)

        stationary_polygon, _ = self.generate_polygons(stationary_actors)
        avoid_obstacle, _ = self.generate_polygons(stationary_actors, self.path_inflation - 0.05)

        graph = nx.Graph()
        self.add_node_to_graph(graph, (*q_init, 0.), avoid_obstacle.values())

        stationary_nodes = self.generate_nodes(stationary_polygon.values())
        for node in stationary_nodes:
            self.add_node_to_graph(graph, node, avoid_obstacle.values())

        self.add_node_to_graph(graph, (*q_goal, 0.), avoid_obstacle.values())

        try:
            nx.shortest_path(graph, source=0, target=len(graph.nodes) -1)
            return False
        except nx.NetworkXNoPath:
            return True

    def add_node_to_graph(self, graph, new_node, polygons, knn=NEAREST_NEIGHBOURS):
        new_node_index = len(graph.nodes)
        graph.add_node(new_node_index, pos=new_node[:2], cost=new_node[2])

        node_connections = 0

        organized_nodes = self.find_closest_nodes(graph, new_node[:2])
        for (node, node_pos) in organized_nodes:
            if node != new_node_index:
                edge_line = LineString([node_pos, new_node[:2]])

                if not any(edge_line.intersects(polygon) for polygon in polygons):
                    graph.add_edge(node, new_node_index, length=edge_line.length)
                    node_connections += 1
            
            if node_connections >= knn:
                break

        return graph

    def generate_polygons(self, actors, overwrite_inflation=None):
        margin = self.path_inflation if overwrite_inflation is None else overwrite_inflation
        
        masses, shapes = {}, {}

        actor_wrappers, actors_state = actors
        for actor in range(1, len(actor_wrappers)):
            actor_wrapper = actor_wrappers[actor]

            mass = actor_wrapper.mass
            name = actor_wrapper.name
            size = actor_wrapper.size

            obs_pos = actors_state[actor, :2]
            obs_rot = quaternion_to_yaw(actors_state[actor, 3:7])

            corners = Polygon([
                (obs_pos[0] - size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] + size[1] / 2),
                (obs_pos[0] - size[0] / 2, obs_pos[1] + size[1] / 2)
            ])

            polygon = Polygon(corners)
            polygon = rotate(polygon, obs_rot, origin=obs_pos, use_radians=True)
            polygon = buffer(polygon, margin, cap_style='flat', join_style='mitre')

            shapes[name] = polygon
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

        obstacles_id_pairs = list(combinations(shapes.keys(), 2))
        for id_1, id_2 in obstacles_id_pairs:
            if masses[id_1] >= self.mass_threshold and masses[id_2] >= self.mass_threshold:
                continue
            
            heavy_id, light_id = [id_1, id_2] if masses[id_1] >= masses[id_2] else [id_2, id_1]
            heavy_ob, light_ob = shapes[heavy_id], shapes[light_id]

            if heavy_ob.distance(light_ob) > (2 * self.path_inflation):
                continue

            prepare(light_ob)
            prepare(heavy_ob)

            passage = nearest_points(light_ob, heavy_ob)
            passage_point = MultiPoint(passage).centroid

            search_distance = self.search_distance_radius(light_ob)
            passages = np.vstack((passages, (passage_point.x, passage_point.y, masses[light_id], search_distance)))

        return passages

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
                    intersection = polygon_i.boundary.intersection(polygon_j.boundary)
                    for point in intersection.geoms:
                        if x_limit[0] <= point.x <= x_limit[1] and y_limit[0] <= point.y <= y_limit[1]:
                            intersection_points.append([point.x, point.y])

        return np.array(intersection_points)

    @staticmethod
    def find_closest_nodes(graph, coordinate):
        nodes_with_distances = []
        for node, pos in graph.nodes(data='pos'):
            distance = np.linalg.norm(np.array(pos) - np.array(coordinate))
            nodes_with_distances.append((node, pos, distance))

        nodes_with_distances.sort(key=lambda x: x[2])

        sorted_nodes = [(node, pos) for node, pos, _ in nodes_with_distances]
        return sorted_nodes

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
                               for dx in np.arange(-radius, radius + 0.01, 0.1)
                               for dy in np.arange(-radius, radius + 0.01, 0.1))
                if is_valid:
                    filtered_nodes.append(node)

        filtered_nodes = np.array(filtered_nodes)
        return np.unique(filtered_nodes, axis=0)

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
    def search_distance_radius(polygon):
        vertices = list(polygon.exterior.coords)
        distance = [LineString([v_1, v_2]).length for v_1, v_2 in combinations(vertices, 2)]
        return max(distance)
