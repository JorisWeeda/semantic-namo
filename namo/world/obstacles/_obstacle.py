
from isaacgym import gymapi

import numpy as np


class Obstacle:

    PATH_ROOT_URDF = "thirdparty/isaacgym/assets"

    def __init__(self, name, size, gym_handle, sim_handle, env_handle):
        self.name = name
        self.size = size
        self.type = None

        self.asset = None
        self.actor = None

        self._gym_handle = gym_handle
        self._sim_handle = sim_handle
        self._env_handle = env_handle

    def build(self):
        asset_options = gymapi.AssetOptions()
        asset_options.density = self.density
        
        self.asset = self._gym_handle.create_box(self._sim_handle, *self.size, asset_options)
        return self.asset

    def asset_to_actor(self, pos, rot):
        rot = [np.deg2rad(rotation) for rotation in rot]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat.from_euler_zyx(*rot)

        object_color = gymapi.Vec3(*self.color)

        self.actor = self._gym_handle.create_actor(self._env_handle, self.asset, pose, self.name)
        self._gym_handle.set_rigid_body_color(self._env_handle, self.actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color)

        return self.actor

    @property
    def pos(self):
        rigid_body_state = self._gym_handle.get_actor_rigid_body_states(self._env_handle, self.actor, gymapi.STATE_ALL)[0]
        x, y, z = rigid_body_state[0][0]['x'], rigid_body_state[0][0]['y'], rigid_body_state[0][0]['z']
        return np.array((x, y, z), dtype=float)

    @property
    def rot(self):
        rigid_body_state = self._gym_handle.get_actor_rigid_body_states(self._env_handle, self.actor, gymapi.STATE_ALL)[0]
        return gymapi.Quat.to_euler_zyx(rigid_body_state[0][1])

    @property
    def mass(self):
        rigid_body_property = self._gym_handle.get_actor_rigid_body_properties(self._env_handle, self.actor)[0]
        return rigid_body_property.mass

    def __hash__(self):
        return hash(self.name)
