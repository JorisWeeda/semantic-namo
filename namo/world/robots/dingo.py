
from isaacgym import gymapi

import numpy as np

from ._robot import Robot

class Dingo(Robot):

    PATH_URDF = "urdf/dingo/dingo.urdf"

    def __init__(self, name, gym_handle, sim_handle, env_handle):
        super().__init__(name, gym_handle, sim_handle, env_handle)
 
    def update_dof_properties(self):
        robot_dof_props = self._gym_handle.get_actor_dof_properties(self._env_handle, self.actor)

        robot_dof_props['stiffness'].fill(1000.0)
        robot_dof_props['damping'].fill(100.0)
        robot_dof_props["driveMode"][0:] = gymapi.DOF_MODE_VEL

        self._gym_handle.set_actor_dof_properties(self._env_handle, self.actor, robot_dof_props)
