
import numpy as np
from isaacgym import gymapi

class Robot:

    PATH_ROOT_URDF = "assets"

    def __init__(self, name, gym_handle, sim_handle, env_handle):
        self.name = name

        self._gym_handle = gym_handle
        self._sim_handle = sim_handle
        self._env_handle = env_handle

        self.asset = None
        self.actor = None

        self.dof_handles = None

    def build(self):
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True

        self.asset = self._gym_handle.load_asset(self._sim_handle, self.PATH_ROOT_URDF, self.PATH_URDF, asset_options)

    def asset_to_actor(self, pos, rot):
        rot = [np.deg2rad(rotation) for rotation in rot]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat.from_euler_zyx(*rot)

        self.actor = self._gym_handle.create_actor(self._env_handle, self.asset, pose, self.name, 0, 1)
        self.update_dof_properties()

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
