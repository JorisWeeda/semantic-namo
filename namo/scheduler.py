

from isaacgym import gymapi

import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from itertools import product
from scipy.ndimage import distance_transform_cdt


from .a_star import a_star


class Scheduler:

    def __init__(self, size, step, goal):
        self.size = size
        self.step = step
        self.goal = goal

    def tasks(self, sim):
                
        rob_size = sim.env_cfg[0].size
        rob_size = [0.5, 0.5, 1.0]

        init_pos = self.get_robot_position(sim)
        goal_pos = self.goal

        obstacles = []

        amount_of_actors = len(sim.env_cfg)
        for actor in range(1, amount_of_actors):
            actor_wrapper = sim.env_cfg[actor]

            pose = self.get_pose(sim, actor)
            mass = self.get_mass(sim, actor)
            name = self.get_name(sim, actor)
            size = actor_wrapper.size
            free = not actor_wrapper.fixed

            obstacles.append({"actor": actor, "name": name, "free": free, "mass": mass, "size": size,  "pos": pose[0], "rot": pose[1]})

        sorted_movable_obstacles = sorted((obstacle for obstacle in obstacles if obstacle["free"]), key=lambda obstacle: obstacle["mass"])

        rob_inflation = [rob_size[0] / 2, rob_size[1] / 2]

        start_grid = self.create_grid(obstacles, rob_inflation)
        robot_path, _  = self.path_planner(start_grid, init_pos, goal_pos)

        transit_tasks = [0, goal_pos]

        if robot_path is not None:
            return transit_tasks
        
        for r in range(1, len(sorted_movable_obstacles)):
            combinations = itertools.combinations(sorted_movable_obstacles, r)

            for combination in list(combinations):
                altered_obstacles = obstacles.copy()   
                
                for obstacle_to_remove in list(combination):
                    altered_obstacles.remove(obstacle_to_remove)             

                altered_grid = self.create_grid(altered_obstacles, rob_inflation)
                robot_path, _  = self.path_planner(altered_grid, init_pos, goal_pos)

                if robot_path is not None:
                    break

            if robot_path is not None:
                break
    
        transfer_tasks = []
        return [[7, [-1.5, 1.5]], [9, [1.5, 0.5]], [0, goal_pos]]
    
    def task_succeeded(self, sim, task, epsilon=5e-2):
        actor, goal = task
        pose = self.get_pose(sim, actor)

        if np.abs(pose[0][0] - goal[0]) < epsilon and np.abs(pose[0][1] - goal[1]) < epsilon:
            return True

        return False


    def path_planner(self, grid, q_rob_init, q_rob_goal):
        rob_init_node_idx, _ = self.nearest_node(q_rob_init)
        rob_goal_node_idx, _ = self.nearest_node(q_rob_goal)

        return a_star(grid[:, :, 2], rob_init_node_idx, rob_goal_node_idx)

    def create_grid(self, obstacles, inflation):
        x, y = self.generate_coordinates()
        xx, yy = np.meshgrid(x, y)

        nodes = np.zeros(xx.shape)

        polygons = self.create_polygons(obstacles, inflation)
        inside_polygons_mask = self.compute_inside_polygons_mask(polygons, xx, yy)

        nodes[inside_polygons_mask] = 1
        tensor = np.stack((xx, yy, nodes), axis=-1)

        return tensor

    def create_polygons(self, obstacles, inflation):
        polygons = []
        for obstacle in obstacles:
            inflated_size_x = obstacle["size"][0] + 2 * inflation[0]
            inflated_size_y = obstacle["size"][1] + 2 * inflation[1]

            corners = np.array([
                [-inflated_size_x / 2, -inflated_size_y / 2],
                [inflated_size_x / 2, -inflated_size_y / 2],
                [inflated_size_x / 2, inflated_size_y / 2],
                [-inflated_size_x / 2, inflated_size_y / 2],
                [-inflated_size_x / 2, -inflated_size_y / 2]
            ])

            rotation_matrix = np.array([
                [np.cos(obstacle["rot"][-1]), -np.sin(obstacle["rot"][-1])],
                [np.sin(obstacle["rot"][-1]), np.cos(obstacle["rot"][-1])]
            ])

            rotated_corners = np.dot(corners, rotation_matrix)
            translated_corners = rotated_corners + np.array(obstacle["pos"][:2])

            polygons.append(Polygon(translated_corners, closed=True))

        return polygons

    def compute_inside_polygons_mask(self, polygons, xx, yy):
        inside_polygons_mask = np.zeros_like(xx, dtype=bool)
        for polygon in polygons:
            inside_polygons_mask |= polygon.contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
        return inside_polygons_mask

    def generate_coordinates(self):
        x = np.arange(-self.size[0] / 2 + self.step, self.size[0] / 2, self.step)
        y = np.arange(-self.size[1] / 2 + self.step, self.size[1] / 2, self.step)
        return x, y

    def nearest_node(self, coordinate):
        x_array, y_array = self.generate_coordinates()
        x_idx, x = self.find_nearest(x_array, coordinate[0])
        y_idx, y = self.find_nearest(y_array, coordinate[1])
        return (x_idx, y_idx), (x, y)

    @staticmethod 
    def find_nearest(array, value):
        idx_nearest = (np.abs(array - value)).argmin()
        return idx_nearest, array[idx_nearest]

    @staticmethod
    def get_pose(sim, actor):
        rigid_body_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        x, y, z = rigid_body_state[0][0]['x'], rigid_body_state[0][0]['y'], rigid_body_state[0][0]['z']
        return np.array((x, y, z), dtype=float), gymapi.Quat.to_euler_zyx(rigid_body_state[0][1])

    @staticmethod
    def get_mass(sim, actor):
        rigid_body_property = sim.gym.get_actor_rigid_body_properties(sim.envs[0], actor)[0]
        return rigid_body_property.mass

    @staticmethod
    def get_name(sim, actor):
        return sim.gym.get_actor_name(sim.envs[0], actor)

    @staticmethod
    def get_robot_position(sim):
        rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        return rob_pos[0].numpy()