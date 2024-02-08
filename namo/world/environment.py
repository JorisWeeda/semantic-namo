
import numpy as np
import random

from isaacgym import gymapi

from .obstacles import Chair, Table, Wall
from .robots import Dingo


class Environment:

    def __init__(self, size, gym_handler, sim_handler, viewer):
        self.gym = gym_handler
        self.sim = sim_handler
        self.view = viewer

        self.size = size

        self.obstacles = {}
        self.robots = {}

    @classmethod
    def build_framework(cls, args):

        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        if args.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.shape_collision_margin = 0.25
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 10

        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 1
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = args.num_threads
            sim_params.physx.use_gpu = args.use_gpu

        else:
            raise RuntimeError(f"Not a valid physics engine: {args.physics_engine}")

        sim_params.use_gpu_pipeline = args.use_gpu
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if sim is None:
            raise RuntimeError(f"Failed to create simulation")
        
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.restitution = 0
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        gym.add_ground(sim, plane_params)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(-6, -6, 5)
        cam_target = gymapi.Vec3(0, 0, 0)

        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        if not viewer:
            raise RuntimeError(f"Failed to create simulation viewer")

        size = args.size
        if not size:
            raise RuntimeError(f"Not a valid environment size given")

        return cls(size, gym, sim, viewer)

    def create_demarcation(self, env, thickness, height):
        width, length = self.size

        b_demarcation_wall = Wall("b-demarcation-wall", (length, thickness, height), self.gym, self.sim, env)
        f_demarcation_wall = Wall("f-demarcation-wall", (length, thickness, height), self.gym, self.sim, env)
        l_demarcation_wall = Wall("l-demarcation-wall", (width, thickness, height), self.gym, self.sim, env)
        r_demarcation_wall = Wall("r-demarcation-wall", (width, thickness, height), self.gym, self.sim, env)

        b_demarcation_wall.build()
        f_demarcation_wall.build()
        l_demarcation_wall.build()
        r_demarcation_wall.build()

        b_demarcation_wall.asset_to_actor((-width/2 - thickness, 0, 0), (0, 0, 90))
        f_demarcation_wall.asset_to_actor((width/2 + thickness, 0, 0), (0, 0, 90))
        l_demarcation_wall.asset_to_actor((0, length/2 + thickness, 0), (0, 0, 0))
        r_demarcation_wall.asset_to_actor((0, -length/2 - thickness, 0), (0, 0, 0))

        self.obstacles["b-demarcation-wall"] = b_demarcation_wall
        self.obstacles["f-demarcation-wall"] = f_demarcation_wall
        self.obstacles["l-demarcation-wall"] = l_demarcation_wall
        self.obstacles["r-demarcation-wall"] = r_demarcation_wall

    def create_obstacle(self, env, category, name, pos, rot):
        width, length = self.size

        pos = pos if pos else (random.uniform(-width/2, width/2), random.uniform(-length/2, length/2), 2)
        rot = rot if rot else (0, 0, random.uniform(0, 360))

        if category == "chair":
            obstacle = Chair(name, self.gym, self.sim, env)
        elif category == "table":
            obstacle = Table(name, self.gym, self.sim, env)
        else:
            raise TypeError(f"Invalid obstacle category given: {category}")

        obstacle.build()
        obstacle.asset_to_actor(pos, rot)
        
        self.obstacles[name] = obstacle

    def create_robot(self, env, category, name, pos, rot):
        if category == "dingo":
            robot = Dingo(name, self.gym, self.sim, env)
        else:
            raise TypeError(f"Invalid robot category given: {category}")

        robot.build()
        robot.asset_to_actor(pos, rot)
        self.robots[name] = robot
        
    def destroy_viewer(self):
        self.gym.destroy_viewer(self.view)
        self.gym.destroy_sim(self.sim)
