from control.mppi_isaac.mppiisaac.utils.conversions import quaternion_to_yaw

import torch


class Objective:
    def __init__(self, u_min, u_max, device):
        self.waypoints = torch.zeros((1, 2), device=device)

        self.u_min = torch.tensor(u_min, device=device)
        self.u_max = torch.tensor(u_max, device=device)

        self.device = device

        self.w_waypoint = 10.0
        self.w_horizon = 5.0
        self.w_rotation = 10.0
        self.w_contact = 5.0
        self.w_control = torch.diag(torch.tensor([1., 1., 1.], device=device))

    def reset(self):
        pass

    def compute_cost(self, sim, u):
        cost_next_waypoint = self._cost_next_waypoint(sim)
        cost_distance = self.w_waypoint * cost_next_waypoint

        cost_targets_ahead = self._cost_targets_ahead(sim)
        cost_horizon = self.w_horizon * cost_targets_ahead

        cost_rotation_to_goal = self._rotation_to_goal(sim)
        cost_rotation = self.w_rotation * cost_rotation_to_goal

        cost_of_contact_force = self._contact_forces_to_goal(sim)
        cost_contacts = self.w_contact * cost_of_contact_force

        cost_controls = self._cost_control_vec(u)

        cost = cost_distance + cost_horizon + cost_rotation + cost_contacts + cost_controls
        return cost

    def _cost_next_waypoint(self, sim):
        goal = self.waypoints[0, :]

        sampled_rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        distance_per_env = torch.linalg.norm(goal - sampled_rob_pos, axis=1)

        distance_norm_per_env = distance_per_env /  (torch.max(distance_per_env) + 1e-3)
        return distance_norm_per_env

    def _cost_targets_ahead(self, sim):
        look_ahead_norm_per_env = torch.zeros((sim.num_envs), device=self.device)

        if self.waypoints.shape[0] > 1:
            sampled_rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
            look_ahead_distance_per_env = torch.cdist(sampled_rob_pos, self.waypoints[1:,  :], p=2)

            look_ahead_norm_per_env = look_ahead_distance_per_env /  (torch.max(look_ahead_distance_per_env) + 1e-3)
            look_ahead_norm_per_env = torch.sum(look_ahead_norm_per_env, dim=1)

        return look_ahead_norm_per_env

    def _rotation_to_goal(self, sim):
        goal = self.waypoints[-1, :]

        tar_yaw = torch.atan2(goal[1] - sim.dof_state[:, 2], goal[0] - sim.dof_state[:, 0])
        rob_yaw = quaternion_to_yaw(sim.rigid_body_state[:, 3, 3:7], self.device)

        rotation_norm_cost_per_env = torch.abs(tar_yaw - rob_yaw) / torch.pi
        return rotation_norm_cost_per_env

    def _contact_forces_to_goal(self, sim):
        net_contact_forces = torch.sum(torch.abs(torch.cat((sim.net_contact_force[:, 0].unsqueeze(1), sim.net_contact_force[:, 1].unsqueeze(1)), 1)), 1)
        number_of_bodies = int(net_contact_forces.size(dim=0) / sim.num_envs)

        reshaped_contact_forces = net_contact_forces.reshape([sim.num_envs, number_of_bodies])
        interaction_per_env = torch.sum(reshaped_contact_forces, dim=1)

        interaction_norm_per_env = interaction_per_env / (torch.max(interaction_per_env) + 1e-3)
        return interaction_norm_per_env
    
    def _cost_control_vec(self, u):
        normalized_u = torch.abs(u / (self.u_max - self.u_min))

        cost_control_vec_per_env = torch.sum((normalized_u @ self.w_control) * normalized_u, dim=1)
        return cost_control_vec_per_env
