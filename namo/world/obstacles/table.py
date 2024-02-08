from ._obstacle import Obstacle


class Table(Obstacle):
    def __init__(self, name, gym_handle, sim_handle, env_handle):
        super().__init__(name, None, gym_handle, sim_handle, env_handle)
        self.size       = (0.8, 1.6, 0.6)
        self.color      = (0.0, 1.0, 0.0) 
        self.density    = 400.0
        self.type       = 'table'