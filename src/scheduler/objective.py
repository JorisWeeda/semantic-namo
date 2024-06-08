from control.mppi_isaac.mppiisaac.utils.conversions import quaternion_to_yaw

import torch


class Objective:
    def __init__(self, u_min, u_max, device):
        self.init = torch.tensor([0., 0., 0., 0., 0., 0.], device=device)
        self.goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=device)
        self.mode = torch.tensor([0., 0.], device=device)

        self.u_min = torch.tensor(u_min, device=device)
        self.u_max = torch.tensor(u_max, device=device)

        self.device = device

        self.w_con = torch.diag(torch.tensor([2., 2., 1.], device=device))
        self.w_dis = 30.0
        self.w_rot = 20.0
        self.w_for = 10.0

    def reset(self):
        pass

    def compute_cost(self, sim, u):
        cost_rotation_to_goal = self._rotation_to_goal(sim)
        cost_distance_to_goal = self._distance_to_goal(sim)
        cost_of_contact_force = self._contact_forces_to_goal(sim)

        cost_controls = self._cost_control_vec(u)

        cost_rotation = self.w_rot * cost_rotation_to_goal
        cost_distance = self.w_dis * cost_distance_to_goal
        cost_contacts = self.w_for * cost_of_contact_force

        return cost_rotation + cost_distance + cost_controls + cost_contacts

    def _cost_control_vec(self, u):
        normalized_u = torch.abs(u / (self.u_max - self.u_min))

        cost_control_vec_per_env = torch.sum(
            (normalized_u @ self.w_con) * normalized_u, dim=1)
        return cost_control_vec_per_env

    def _rotation_to_goal(self, sim):
        tar_yaw = torch.atan2(self.goal[1] - sim.dof_state[:, 2], self.goal[0] - sim.dof_state[:, 0])
        rob_yaw = quaternion_to_yaw(sim.rigid_body_state[:, 3, 3:7], self.device)

        rotation_norm_cost_per_env = torch.abs(tar_yaw - rob_yaw) / torch.pi
        return rotation_norm_cost_per_env

    def _distance_to_goal(self, sim):
        sampled_rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        initial_rob_pos = torch.tensor([self.init[0], self.init[2]], device=self.device)

        initial_distance = torch.linalg.norm(initial_rob_pos - self.goal[:2])
        distance_per_env = torch.linalg.norm(sampled_rob_pos - self.goal[:2], axis=1)

        distance_norm_per_env = distance_per_env / (initial_distance + 1e-3)
        torch.clamp(distance_norm_per_env, min=0.5)
        return distance_norm_per_env

    def _contact_forces_to_goal(self, sim):
        net_contact_forces = torch.sum(torch.abs(torch.cat(
            (sim.net_contact_force[:, 0].unsqueeze(1), sim.net_contact_force[:, 1].unsqueeze(1)), 1)), 1)
        number_of_bodies = int(net_contact_forces.size(dim=0) / sim.num_envs)

        reshaped_contact_forces = net_contact_forces.reshape(
            [sim.num_envs, number_of_bodies])
        interaction_per_env = torch.sum(reshaped_contact_forces, dim=1)

        interaction_norm_per_env = interaction_per_env / (torch.max(interaction_per_env) + + 1e-3)
        return interaction_norm_per_env
