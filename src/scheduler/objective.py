from isaacgym import gymapi

import torch
import quaternionic

import numpy as np

from tf.transformations import euler_from_quaternion
from semantic_namo.cfg import DingoCostConfig


class Objective:
    def __init__(self):
        self.mode = torch.tensor([0., 0.])   # [transit(0)/transfer(1), block_index]
        self.goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

        self.w_con = np.diag([1.0, 1.0, 1.0])
        self.w_dis = 200.0
        self.w_rot = 5.0
        self.w_int = 1.0

    def compute_cost(self, sim, u):
        cost_distance_to_goal = self.w_dis * self._distance_to_goal(sim)
        cost_rotation_to_goal = self.w_rot * self._rotation_to_goal(sim)
        cost_interact_to_goal = self.w_int * self._interact_to_goal(sim)

        cost_input_weight_mat = torch.sum((u @ self.w_con) * u, dim=1)

        total_cost = cost_distance_to_goal + \
                     cost_rotation_to_goal + \
                     cost_interact_to_goal + \
                     cost_input_weight_mat
        
        return total_cost

    def _distance_to_goal(self, sim):
        rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)

        distance_per_env =  torch.linalg.norm(rob_pos - self.goal[:2], axis=1)
        return distance_per_env

    def _rotation_to_goal(self, sim):
        goal_quaternion = quaternionic.array(self.goal[3:])
        envs_quaternion = quaternionic.array(sim.rigid_body_state[:, 3, 3:7])

        rotation_per_env = quaternionic.distance.rotation.intrinsic(envs_quaternion, goal_quaternion)
        return rotation_per_env

    def _interact_to_goal(self, sim):
        net_contact_forces = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)), 1)
        reshaped_contact_forces = net_contact_forces.reshape([sim.num_envs, sim.num_bodies])

        collision_cost_per_env = torch.sum(reshaped_contact_forces, dim=1)
        return collision_cost_per_env