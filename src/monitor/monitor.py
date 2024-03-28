from isaacgym import gymapi

import torch

import numpy as np
import matplotlib.pyplot as plt


class Monitor:
    def __init__(self, actors_mass, actors_name):
        self.actors_mass = actors_mass
        self.actors_name = actors_name
        
        self.force_arr = np.empty((0, len(actors_mass) + 1), dtype=float)
        self.veloc_arr = np.empty((0, len(actors_mass) + 1), dtype=float)
        self.accel_arr = np.empty((0, len(actors_mass) + 1), dtype=float)
        self.loads_arr = np.empty((0, len(actors_mass) + 1), dtype=float)

    @classmethod
    def create_robot_monitor(cls, sim):
        actors_mass = {}
        actors_name = {}

        amount_of_actors = len(sim.env_cfg)
        for actor in range(0, amount_of_actors):
            mass = cls.get_mass(sim, actor)
            name = cls.get_name(sim, actor)

            if sim.gym.get_actor_force_sensor_count(sim.envs[0], actor):
                actors_mass[actor] = mass
                actors_name[actor] = name
        
        return cls(actors_mass, actors_name)

    def add_data(self, sim):
        timestamp = sim.gym.get_elapsed_time(sim.sim)

        if timestamp > 1.0:
            force = [self.get_force(sim, actor) for actor in self.actors_mass.keys()]
            self.force_arr = np.vstack((self.force_arr, [timestamp, *force]))

            veloc = [self.get_lin_vel(sim, actor).sum() for actor in self.actors_mass.keys()]
            self.veloc_arr = np.vstack((self.veloc_arr, [timestamp, *veloc]))

            if self.veloc_arr.shape[0] > 2:
                delta_v = self.veloc_arr[-1, 1:] - self.veloc_arr[-2, 1:]
                delta_t = self.veloc_arr[-1, 0] - self.veloc_arr[-2, 0]

                accel = delta_v / delta_t
                loads = [force[i] / accel[i] if accel[i] > 0.0 else 0.0 for i in range(len(force))]

                self.accel_arr = np.vstack((self.accel_arr, [timestamp, *accel]))
                self.loads_arr = np.vstack((self.loads_arr, [timestamp, *loads]))
    
    def plotter(self):
        fig, axis = plt.subplots(2, 2, figsize=(14, 6))

        for idx, name in enumerate(self.actors_name.values()):
            axis[0][0].plot(self.veloc_arr[:, 0], self.veloc_arr[:, idx + 1], label=name)

        axis[0][0].set_xlabel('timestamp')
        axis[0][0].set_ylabel('Velocity')
        axis[0][0].grid(True)  
        axis[0][0].legend()

        for idx, name in enumerate(self.actors_name.values()):
            axis[1][0].plot(self.accel_arr[:, 0], self.accel_arr[:, idx + 1], label=name)

        axis[1][0].set_xlabel('timestamp')
        axis[1][0].set_ylabel('Acceleration')
        axis[1][0].grid(True)  
        axis[1][0].legend()

        for idx, name in enumerate(self.actors_name.values()):
            axis[0][1].plot(self.force_arr[:, 0], self.force_arr[:, idx + 1], label=name)

        axis[0][1].set_xlabel('timestamp')
        axis[0][1].set_ylabel('Force')
        axis[0][1].grid(True)  
        axis[0][1].legend()

        for idx, name in enumerate(self.actors_name.values()):
            axis[1][1].plot(self.loads_arr[:, 0], self.loads_arr[:, idx + 1], label=name)

        axis[1][1].set_xlabel('timestamp')
        axis[1][1].set_ylabel('Mass')
        axis[1][1].grid(True)  
        axis[1][1].legend()

        plt.show()

    @staticmethod
    def get_force(sim, actor):
        sensor_data = sim.gym.get_actor_force_sensor(sim.envs[0], actor, 0).get_forces()
        return sensor_data.force.length()

    @staticmethod
    def get_mass(sim, actor):
        rigid_body_property = sim.gym.get_actor_rigid_body_properties(sim.envs[0], actor)[0]
        return rigid_body_property.mass

    @staticmethod
    def get_name(sim, actor):
        return sim.gym.get_actor_name(sim.envs[0], actor)

    @staticmethod
    def get_pos(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return np.array([rb_state["pose"]["p"]["x"], rb_state["pose"]["p"]["y"], rb_state["pose"]["p"]["z"]], dtype=np.float32)

    @staticmethod
    def get_ori(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return gymapi.Quat.to_euler_zyx(rb_state["pose"]["r"])

    @staticmethod
    def get_lin_vel(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return np.array([rb_state["vel"]["linear"]["x"], rb_state["vel"]["linear"]["y"], rb_state["vel"]["linear"]["z"]], dtype=np.float32)

    @staticmethod
    def get_ang_vel(sim, actor):
        rb_state = sim.gym.get_actor_rigid_body_states(sim.envs[0], actor, gymapi.STATE_ALL)[0]
        return np.array([rb_state["vel"]["angular"]["x"], rb_state["vel"]["angular"]["y"], rb_state["vel"]["angular"]["z"]], dtype=np.float32)
        
    @staticmethod
    def get_robot_position(sim):
        rob_pos = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1)), 1)
        return rob_pos[0].numpy()
    
    @staticmethod
    def get_robot_velocity(sim):
        rob_vel = torch.cat((sim.dof_state[:, 1].unsqueeze(1), sim.dof_state[:, 3].unsqueeze(1)), 1)
        return rob_vel[0].numpy()
    
    