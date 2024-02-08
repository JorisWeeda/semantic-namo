
import hydra
import numpy as np
import threading
import yaml

from isaacgym import gymapi, gymutil
from motion_planning.mppiisaac.utils.config_store import ExampleConfig

from namo.control import ManualControl
from namo.planning import Planner
from namo.world import Environment
from namo.dashboard.dashboard import Dashboard


# Simulation parameters
world_config = "assets/worlds/sandbox.yaml"
with open(world_config, "r") as stream:
    params = yaml.safe_load(stream)

# Create isaacgym environment
args = gymutil.parse_arguments(description=params["description"],
                               custom_parameters=[{"name": "--size", "type": list, "default": params["framework"]["size"], "help": "Environment size"}])
environment = Environment.build_framework(args)

gym = environment.gym
sim = environment.sim
viewer = environment.view

env_lower = gymapi.Vec3(*params["framework"]["env_lower"])
env_upper = gymapi.Vec3(*params["framework"]["env_upper"])
number_of_environments = params["framework"]["amount_of_envs"]

env = gym.create_env(sim, env_lower, env_upper, number_of_environments)

# Populate isaacgym environment
thickness   = params["environment"]["demarcation"]["thickness"]
height      = params["environment"]["demarcation"]["height"]
size        = params["framework"]["size"]

environment.create_demarcation(env, thickness, height)

for obstacle in params["environment"]["obstacles"]:
    obs_category = next(iter(obstacle))
    obs_params = obstacle[obs_category]
    
    name = obs_params["name"]
    pos = obs_params.get("pos", None)
    rot = obs_params.get("rot", None)

    environment.create_obstacle(env, obs_category, name, pos, rot)

for robot in params["environment"]["robots"]:
    rob_category = next(iter(robot))
    rob_params = robot[rob_category]

    name = rob_params["name"]
    pos = rob_params.get("pos", None)
    rot = rob_params.get("rot", None)
    
    environment.create_robot(env, rob_category, name, pos, rot)

# Create planners 
brim = params["planner"]["brim"]
step = params["planner"]["step"]
goal = params["planner"]["goal"]
planner = Planner(gym, env, environment.size, step)

# Add manual control for the robot base
dingo_robot = environment.robots["Robot 1"]
manual_control = ManualControl.link_mecanum_control(gym, env, dingo_robot)

# Subscribe to events for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "planning")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "stop")

initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

allow_running = True
while allow_running and not gym.query_viewer_has_closed(environment.view):
    t = gym.get_sim_time(sim)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

        if evt.action == "motion" and evt.value > 0:
            manual_control.move_mecanum(2, 2, 0)

        if evt.action == "planning" and evt.value > 0:
            planner.paths(environment.obstacles, dingo_robot, goal)
        
        if evt.action == "stop" and evt.value > 0:
            allow_running = False

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(environment.view, sim, True)

    gym.sync_frame_time(sim)

environment.destroy_viewer()
