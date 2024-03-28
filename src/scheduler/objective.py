import torch

import pytorch3d.transforms as p3dtransforms
from dynamic_reconfigure.server import Server

from semantic_namo.cfg import DingoCostConfig  #type: ignore


class Objective:
    def __init__(self):
        self._parameter_server = Server(DingoCostConfig, self.update_weights) 

        self._mode = torch.tensor([0., 0.])   # [transit(0)/transfer(1), block_index]
        self._goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

        self.w_nav = 1.0
        self.w_obs = 0.1
        self.w_coll = 0.1

        # Tuning of the weights for box
        self.w_robot_to_block_pos = 0.2
        self.w_block_to_goal_pos = 2.
        self.w_block_to_goal_ort = 3.0
        self.w_push_align = 0.6
        self.w_collision = 10
        self.w_vel = 0.

    def update_weights(self, config, *args):
        self.w_robot_to_block_pos= config["w_robot_to_block_pos"]#1.0#10#2
        self.w_block_to_goal_pos=  config["w_block_to_goal_pos"]#2.5#25#12.0 
        self.w_block_to_goal_ort=  config["w_block_to_goal_ort"]#2.1#21#10.0
        self.w_push_align=         config["w_push_align"]#1.5#15#4.2
        self.w_collision=          config["w_collision"]#0.0
        self.w_vel=                config["w_vel"]#0.0

        print('updated weights')
        print(config["w_robot_to_block_pos"])
        print(config["w_block_to_goal_pos"])
        print(config["w_block_to_goal_ort"])
        print(config["w_push_align"])
        print(config["w_collision"])
        print(config["w_vel"])
        print('\n')
        return config

    def compute_cost(self, sim):
        pos = self._get_robot_position(sim)

        if self._mode[0] == 0.:
            cost = self._compute_navigation_cost(pos)

            if len(sim.obstacle_indices) != 0:
                cost += self._compute_collision_cost(sim)
                cost += self._compute_obstacle_cost(sim, pos)

            return cost
  
        elif self._mode[0] == 1.:
            return self._compute_transfer_cost(sim, pos)

        else:
            raise RuntimeError

    def _get_robot_position(self, sim):
        return torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)

    def _compute_navigation_cost(self, pos):
        nav_cost = torch.linalg.norm(pos - self._goal[:2], axis=1)
        return nav_cost * self.w_nav

    def _compute_collision_cost(self, sim):
        xy_contacts = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)), 1)
        coll = torch.sum(xy_contacts.reshape([sim.num_envs, int(xy_contacts.size(dim=0) / sim.num_envs)])[:, 1:sim.num_bodies], 1)
        return coll * self.w_coll

    def _compute_obstacle_cost(self, sim, pos):
        obs_cost = torch.sum(1 / torch.linalg.norm(sim.obstacle_positions[:, :, :2] - pos.unsqueeze(1), axis=2), axis=1)
        return obs_cost * self.w_obs

    def _compute_transfer_cost(self, sim, pos):
        block_index = int(self._mode[1])

        block_pos = sim.root_state[:, block_index, :3]
        block_ort = sim.root_state[:, block_index, 3:7]
        block_vel = sim.root_state[:, block_index, 7:10]

        block_goal_pose = torch.clone(self._goal)
        block_ort_goal = torch.clone(block_goal_pose[3:7])
        goal_yaw = torch.atan2(2.0 * (block_ort_goal[-1] * block_ort_goal[2] + block_ort_goal[0] * block_ort_goal[1]), block_ort_goal[-1] * block_ort_goal[-1] + block_ort_goal[0] * block_ort_goal[0] - block_ort_goal[1] * block_ort_goal[1] - block_ort_goal[2] * block_ort_goal[2])
        
        robot_to_block = pos - block_pos[:, 0:2]
        block_to_goal = block_goal_pose[0:2] - block_pos[:, 0:2]
    
        block_yaws = torch.atan2(2.0 * (block_ort[:, -1] * block_ort[:, 2] + block_ort[:, 0] * block_ort[:, 1]), block_ort[:, -1] * block_ort[:, -1] + block_ort[:, 0] * block_ort[:, 0] - block_ort[:, 1] * block_ort[:, 1] - block_ort[:, 2] * block_ort[:, 2])
        block_yaws = p3dtransforms.matrix_to_euler_angles(p3dtransforms.quaternion_to_matrix(block_ort), "ZYX")[:, -1]
    
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis=1)
        block_to_goal_pos = torch.linalg.norm(block_to_goal, axis=1)
        block_to_goal_ort = torch.abs(block_yaws - goal_yaw)

        push_align = torch.sum(robot_to_block[:, 0:2] * block_to_goal, 1) / (robot_to_block_dist * block_to_goal_pos) + 1

        vel = torch.linalg.norm(block_vel, axis=1)

        xy_contacts = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)), 1)
        coll = torch.sum(xy_contacts.reshape([sim.num_envs, int(xy_contacts.size(dim=0) / sim.num_envs)])[:, (sim.num_bodies - 1):sim.num_bodies], 1)  

        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_dist
            + self.w_block_to_goal_pos * block_to_goal_pos
            + self.w_block_to_goal_ort * block_to_goal_ort
            + self.w_push_align * push_align
            + self.w_collision * coll
            + self.w_vel * vel
        )

        return total_cost

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, new_mode):
        print(f"Objective | new mode set : {new_mode}")
        self._mode = new_mode

    @property
    def goal(self):
        return self._goal
    
    @goal.setter
    def goal(self, new_goal):
        print(f"Objective | new goal set : {new_goal}")
        self._goal = new_goal
