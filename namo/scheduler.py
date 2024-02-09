

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from itertools import product

from .a_star import a_star


class Scheduler:


    @staticmethod
    def tasks(sim, size, step, goal):
        print(f"sim.root_state: {sim.root_state.shape}")
        _, amount_of_actors, _ = sim.root_state.shape

        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)

        print(f'dof_state: {dof_state}')

        rob_pose = sim.root_state[:, 4, :3]
        
        print(f'rob_pose: {rob_pose}')
        for obs_idx in range(1, amount_of_actors):
            obs_pose = sim.root_state[:, obs_idx, :3]
            print(f'obs_pose {obs_idx}: {obs_pose}')



    def paths(self, obstacles, robot, q_goal):

        sorted_obstacles = sorted(obstacles.values(), key=lambda x: x.mass)
        disc_config_grid = self.create_grid(sorted_obstacles)

        transit_path, _  = self.transit_planner(disc_config_grid, robot.pos, q_goal)

        if not transit_path:
            print("attempt to find initial path failed, trying to remove obstacles")

            updated_sorted_obstacles = sorted_obstacles
            for idx, obstacle in enumerate(sorted_obstacles):

                print(f"Removing obstacle: {obstacle.name})")
                updated_sorted_obstacles.pop(idx)

                disc_config_grid = self.create_grid(updated_sorted_obstacles)
                transit_path, _  = self.transit_planner(disc_config_grid, robot.pos, q_goal)

                if transit_path:
                    break

        x_coords = disc_config_grid[:, :, 0].flatten()
        y_coords = disc_config_grid[:, :, 1].flatten()
        node_values = disc_config_grid[:, :, 2].flatten()

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for x_coords, y_coords, and node_values
        ax1 = axes[0]
        ax1.scatter(x_coords, y_coords, c=node_values, cmap='binary', edgecolors='k')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Grid Nodes Visualization')
        ax1.grid(True)

        # Plot for path_to_goal
        ax2 = axes[1]

        for coord in transit_path:
            x, y = disc_config_grid[coord[0], coord[1], :2] 
            ax2.scatter(x, y, color='green')

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Path to Goal')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        return transit_path

    def transit_planner(self, grid, q_rob_init, q_rob_goal):
        rob_init_node_idx, _ = self.nearest_node(q_rob_init)
        rob_goal_node_idx, _ = self.nearest_node(q_rob_goal)

        return a_star(grid[:, :, 2], rob_init_node_idx, rob_goal_node_idx)

    def create_grid(self, obstacles):
        x, y = self.generate_coordinates()
        xx, yy = np.meshgrid(x, y)

        nodes = np.zeros(xx.shape)

        polygons = self.create_polygons(obstacles)
        inside_polygons_mask = self.compute_inside_polygons_mask(polygons, xx, yy)

        nodes[inside_polygons_mask] = 1
        tensor = np.stack((xx, yy, nodes), axis=-1)

        return tensor

    def generate_coordinates(self):
        x = np.arange(-self.size[0] / 2 + self.step, self.size[0] / 2, self.step)
        y = np.arange(-self.size[1] / 2 + self.step, self.size[1] / 2, self.step)

        return x, y

    def create_polygons(self, obstacles):
        polygons = []
        for obstacle in obstacles:
            corners = np.array([
                [-obstacle.size[0] / 2, -obstacle.size[1] / 2],
                [obstacle.size[0] / 2, -obstacle.size[1] / 2],
                [obstacle.size[0] / 2, obstacle.size[1] / 2],
                [-obstacle.size[0] / 2, obstacle.size[1] / 2],
                [-obstacle.size[0] / 2, -obstacle.size[1] / 2]
            ])

            rotation_matrix = np.array([
                [np.cos(obstacle.rot[-1]), -np.sin(obstacle.rot[-1])],
                [np.sin(obstacle.rot[-1]), np.cos(obstacle.rot[-1])]
            ])

            rotated_corners = np.dot(corners, rotation_matrix)
            translated_corners = rotated_corners + np.array(obstacle.pos[:2])

            polygons.append(Polygon(translated_corners, closed=True, color=obstacle.color))

        return polygons

    def compute_inside_polygons_mask(self, polygons, xx, yy):
        inside_polygons_mask = np.zeros_like(xx, dtype=bool)
        for polygon in polygons:
            inside_polygons_mask |= polygon.contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

        return inside_polygons_mask

    def nearest_node(self, coordinate):
        x_array, y_array = self.generate_coordinates()
        x_idx, x = self.find_nearest(x_array, coordinate[0])
        y_idx, y = self.find_nearest(y_array, coordinate[1])
        return (x_idx, y_idx), (x, y)

    @staticmethod 
    def find_nearest(array, value):
        idx_nearest = (np.abs(array - value)).argmin()
        return idx_nearest, array[idx_nearest]
