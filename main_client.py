from motion.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
from motion.mppiisaac.utils.config_store import ExampleConfig

import io
import os
import hydra
import numpy as np
import torch
import yaml
import zerorpc

from omegaconf import OmegaConf
from isaacgym import gymapi
from scipy.spatial.transform import Rotation

from namo import Scheduler


def set_viewer(gym, viewer, position, target):
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))


def torch_to_bytes(torch_tensor) :
    buff = io.BytesIO()
    torch.save(torch_tensor, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(buffer):
    buff = io.BytesIO(buffer)
    return torch.load(buff)


@hydra.main(version_base=None, config_path="config", config_name="config_heijn_push")
def client(cfg: ExampleConfig):
    cfg = OmegaConf.to_object(cfg)

    actors=[]
    for actor_name in cfg["actors"]:
        with open(f'{os.getcwd()}/config/actors/{actor_name}.yaml') as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=yaml.SafeLoader)))

    sim = IsaacGymWrapper(
        cfg["isaacgym"],
        init_positions=cfg["initial_actor_positions"],
        actors=actors,
        num_envs=1,
        viewer=True,
        device=cfg["mppi"].device,
    )

    # Simulation parameters
    world_config = "config/worlds/blockage.yaml"
    with open(world_config, "r") as stream:
        params = yaml.safe_load(stream)

    # Create custom environment of a single simulated environment 
    size = params["environment"]["size"]
    additions =[]
    
    for wall in params["environment"]["demarcation"]:
        obs_type = next(iter(wall))
        obs_args = params["objects"][obs_type]

        obstacle = {**obs_args, **wall[obs_type]}
        
        rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
        obstacle["init_ori"] = list(rot)

        additions.append(obstacle)

    for obstacle in params["environment"]["obstacles"]:
        obs_type = next(iter(obstacle))
        obs_args = params["objects"][obs_type]

        obstacle = {**obs_args, **obstacle[obs_type]}

        rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
        obstacle["init_ori"] = list(rot)
        
        additions.append(obstacle)

    sim.add_to_envs(additions)

    # Robot variables
    init_pos = params["environment"]["robot"]["init_pos"]
    init_vel = params["environment"]["robot"]["init_vel"]

    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2]], device=sim.device))

    # Camera variables
    cam_pos = params["camera"]["pos"]
    cam_tar = params["camera"]["tar"]
    set_viewer(sim.gym, sim.viewer, cam_pos, cam_tar)

    # Connect to the planner
    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    planner.add_to_env(additions)

    # Set objective function and mode for the mobile robot
    goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=cfg["mppi"].device)
    mode = torch.tensor([0., 0.], device=cfg["mppi"].device)

    planner.update_objective_goal(torch_to_bytes(goal))
    planner.update_objective_mode(torch_to_bytes(mode))

    # Scheduler
    step = params["scheduler"]["step"]
    goal = params["scheduler"]["goal"]

    scheduler = Scheduler(size, step, goal)

    # Copy initial state
    initial_state = np.copy(sim.gym.get_sim_rigid_body_states(sim.sim, gymapi.STATE_ALL))

    # Subscribe to keyboard events
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_ESCAPE, "stop")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_S, "scheduler")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_A, "action")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_V, "viewer")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_R, "reset")

    # Running variables
    is_allowed_to_run = True
    is_viewer_running = False

    # tasks
    ready = True
    tasks = None
    stage = 0

    while is_allowed_to_run and not is_viewer_running:
        is_viewer_running = sim.gym.query_viewer_has_closed(sim.viewer)

        for evt in sim.gym.query_viewer_action_events(sim.viewer):
            if evt.action == "stop" and evt.value > 0:
                is_allowed_to_run = False
            
            if evt.action == "reset" and evt.value > 0:
                sim.gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
            
            if evt.action == "viewer" and evt.value > 0:
                set_viewer(sim.gym, sim.viewer, cam_pos, cam_tar)

            if evt.action == "scheduler" and evt.value > 0:
                tasks = scheduler.tasks(sim)

        planner.reset_rollout_sim(torch_to_bytes(sim.dof_state[0]),
                                  torch_to_bytes(sim.root_state[0]),
                                  torch_to_bytes(sim.rigid_body_state[0]))

        if tasks is not None:
            if ready and len(tasks) != stage:
                actor, [x, y] = tasks[stage]
                move_mode = 0. if actor == 0 else 1.
                
                goal = torch.tensor([x, y, 0., 0., 0., 0., 1.], device=cfg["mppi"].device)
                mode = torch.tensor([move_mode, actor], device=cfg["mppi"].device)

                planner.update_objective_goal(torch_to_bytes(goal))
                planner.update_objective_mode(torch_to_bytes(mode))
    
                ready = False

            action = bytes_to_torch(planner.command())

            if torch.any(torch.isnan(action)):
                print("nan action")
                action = torch.zeros_like(action)

            sim.set_dof_velocity_target_tensor(action)

            if len(tasks) != stage and scheduler.task_succeeded(sim, tasks[stage]):
                ready = True
                stage += 1

        sim.step()


if __name__ == "__main__":
    client()

