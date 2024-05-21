import numpy as np
import networkx as nx

from shapely.geometry import Point, LineString, Polygon
from control.mppi_isaac.mppiisaac.utils.conversions import quaternion_to_yaw


class FailedToGeneratePath(Exception):
    """Base class to generate custom exception if generating path from edges."""

    def __init__(self, error_message):
        """Construct custom error with custom error message
        :param error_message: The custom error message
        """
        super().__init__(error_message)


class RRT:

    MAX_ITERATIONS = int(1e6)

    def __init__(self, range_x, range_y, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.path_inflation = path_inflation

    def graph(self, q_init, q_goal, actors, intermediate_goal_check=True):
        c_space = self.generate_polygons(actors)
        
        q_init = np.array(q_init)
        q_goal = np.array(q_goal)

        if self.is_node_in_obstacle_space(q_init, c_space):
            raise FailedToGeneratePath("Given initial configuration is in obstacle space.")

        if self.is_node_in_obstacle_space(q_goal, c_space):
            raise FailedToGeneratePath("Given goal configuration is in obstacle space.")

        nodes = np.array([q_init])  # Initialize nodes with the initial node
        edges = []

        for _ in range(self.MAX_ITERATIONS):
            new_node = self.generate_random_node()

            if any((new_node == node).all() for node in nodes):
                continue

            if self.is_node_in_obstacle_space(new_node, c_space):
                continue

            closest_node = self.closest_node(new_node, nodes)
            if self.is_edge_in_obstacle_space(closest_node, new_node, c_space):
                continue

            nodes = np.vstack((nodes, new_node))

            closest_node_idx = np.where((nodes == closest_node).all(axis=1))[0][0]
            created_node_idx = np.where((nodes == new_node).all(axis=1))[0][0]

            edges.append((closest_node_idx, created_node_idx))

            if intermediate_goal_check and self.is_goal_reachable(q_goal, nodes, c_space):
                closest_node = self.closest_node(q_goal, nodes)
                closest_node_idx = np.where((nodes == closest_node).all(axis=1))[0][0]

                nodes = np.vstack((nodes, q_goal))
                created_node_idx = np.where((nodes == q_goal).all(axis=1))[0][0]

                edges.append((closest_node_idx, created_node_idx))
                break
            
        if q_goal.tolist() not in nodes.tolist():
            return None
        
        edges = np.array(edges)
        
        graph = nx.Graph()

        for node_idx, node in enumerate(nodes):
            graph.add_node(node_idx, pos=(node[0], node[1]), cost=0.)

        for _, (start_idx, end_idx) in enumerate(edges):
            edge_length = np.linalg.norm(nodes[start_idx] - nodes[end_idx])
            graph.add_edge(start_idx, end_idx, length=edge_length)
        
        return graph

    def generate_random_node(self):
        rand_x = np.random.uniform(*self.range_x)
        rand_y = np.random.uniform(*self.range_y)
        return np.array([rand_x, rand_y])

    def generate_polygons(self, actors, overwrite_inflation=None):
        overwrite_inflation = self.path_inflation if overwrite_inflation is None else overwrite_inflation
        
        shapes = []
        actor_wrappers, actors_state = actors
        for actor in range(1, len(actor_wrappers)):
            actor_wrapper = actor_wrappers[actor]

            obs_pos = actors_state[actor, :2]
            obs_rot = quaternion_to_yaw(actors_state[actor, 3:7])

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

            shapes.append(Polygon(translate_corners)) 
        return shapes

    @staticmethod
    def is_goal_reachable(q_goal, nodes, c_space_obstacles):
        closest_node = RRT.closest_node(q_goal, nodes)

        if RRT.is_edge_in_obstacle_space(q_goal, closest_node, c_space_obstacles):
            return False

        return True

    @staticmethod
    def is_edge_in_obstacle_space(node_i, node_j, c_space_obstacles):
        edge = LineString([node_i, node_j])
        return any(edge.intersects(obstacle) for obstacle in c_space_obstacles)

    @staticmethod
    def is_node_in_obstacle_space(node, c_space_obstacles):
        return any(obstacle.contains(Point(*node)) for obstacle in c_space_obstacles)

    @staticmethod
    def closest_node(current_node, all_nodes):
        distances = np.linalg.norm(all_nodes - current_node, axis=1)
        return all_nodes[np.argmin(distances)]