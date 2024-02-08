
import numpy as np

class ManualControl:

    MECANUM_DOF_NAMES = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
    
    def __init__(self, gym_handle, env_handle, robot_actor, dof_handles):
        self.robot_actor = robot_actor
        self.dof_handles = dof_handles

        self._gym_handle = gym_handle
        self._env_handle = env_handle

    @classmethod
    def link_mecanum_control(cls, gym_handle, env_handle, robot):

        dof_handles = []
        for dof_name in cls.MECANUM_DOF_NAMES:
            dof_handles.append(gym_handle.find_actor_dof_handle(env_handle, robot.actor, dof_name))

        return cls(gym_handle, env_handle, robot.actor, dof_handles)

    def move_mecanum(self, x_linear, y_linear, z_angular):
        fl_wheel = x_linear + y_linear + z_angular
        fr_wheel = x_linear - y_linear - z_angular
        rl_wheel = x_linear - y_linear + z_angular
        rr_wheel = x_linear + y_linear - z_angular

        self._gym_handle.set_dof_target_velocity(self._env_handle, self.dof_handles[0], fl_wheel)
        self._gym_handle.set_dof_target_velocity(self._env_handle, self.dof_handles[1], fr_wheel)
        self._gym_handle.set_dof_target_velocity(self._env_handle, self.dof_handles[2], rl_wheel)
        self._gym_handle.set_dof_target_velocity(self._env_handle, self.dof_handles[3], rr_wheel)
        
        
        
        

    
