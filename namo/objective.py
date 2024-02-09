
from isaacgym import gymapi

import torch
import pytorch3d.transforms

class Objective:
    def __init__(self, device):
        self._mode = torch.tensor([0., 0., 0.], device=device)   # [transit(0)/transfer(1), robot_index, block_index]
        self._goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=device)

        self.w_nav = 1.0
        self.w_obs = 1.0
        self.w_coll = 0.1

    def compute_cost(self, sim):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.linalg.norm(pos - self._goal[:2], axis=1)
        obs_cost = torch.sum(1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),axis=1)

        xy_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)),1)
        coll = torch.sum(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])[:, 1:sim.num_bodies], 1) # skip the first, it is the robot

        return nav_cost * self.w_nav + coll * self.w_coll # + obs_cost * self.w_obs

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
