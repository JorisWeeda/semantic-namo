from motion.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
from motion.mppiisaac.utils.config_store import ExampleConfig

import io
import os
import hydra
import numpy as np
import time
import torch
import yaml
import zerorpc

from omegaconf import OmegaConf
from isaacgym import gymapi
from scipy.spatial.transform import Rotation

from namo import Scheduler
from monitor import Monitor

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


@hydra.main(version_base=None, config_path="config", config_name="config_dingo_push")
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
        viewer=False,
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

    # Running variables
    is_allowed_to_run = True
    is_task_scheduled = False

    # tasks
    ready = True
    tasks = None
    stage = 0

    # Plotters
    monitor = Monitor.create_robot_monitor(sim)
    
    while is_allowed_to_run:

        if not is_task_scheduled and sim.gym.get_elapsed_time(sim.sim) > 1.0:
            tasks = scheduler.tasks(sim)
            is_task_scheduled = True

        planner.reset_rollout_sim(torch_to_bytes(sim.dof_state[0]),
                                  torch_to_bytes(sim.root_state[0]),
                                  torch_to_bytes(sim.rigid_body_state[0]))

        if tasks is not None:
            if ready and stage < len(tasks):
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

            if stage < len(tasks) and scheduler.task_succeeded(sim, tasks[stage]):
                print(f"Tasked succeeded, elapsed time: {sim.gym.get_elapsed_time(sim.sim)}")
                ready = True
                stage += 1
            
            if stage == len(tasks):
                break
        
        monitor.add_data(sim)

        sim.step()

    sim.gym.destroy_viewer(sim.viewer)
    sim.gym.destroy_sim(sim.sim)

    monitor.plotter()

if __name__ == "__main__":
    client()

