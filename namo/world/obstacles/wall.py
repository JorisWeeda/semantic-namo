from isaacgym import gymapi

from ._obstacle import Obstacle


class Wall(Obstacle):

    def __init__(self, name, size, gym_handle, sim_handle, env_handle):
        super().__init__(name, size, gym_handle, sim_handle, env_handle)
        self.color      = (0.0, 0.2, 0.4)
        self.density    = 1000.0
        self.type       = 'wall'

    def build(self):
        asset_options = gymapi.AssetOptions()
        asset_options.density = self.density
        asset_options.fix_base_link = True

        self.asset = self._gym_handle.create_box(self._sim_handle, *self.size, asset_options)
        return self.asset
