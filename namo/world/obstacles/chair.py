
from ._obstacle import Obstacle


class Chair(Obstacle):
    def __init__(self, name, gym_handle, sim_handle, env_handle):
        super().__init__(name, None, gym_handle, sim_handle, env_handle)
        self.size       = (0.5, 0.5, 0.8)
        self.color      = (1.0, 0.0, 0.0) 
        self.density    = 10.0
        self.type       = 'chair'