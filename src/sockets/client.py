
import control.mppiisaac.planner.isaacgym_wrapper      # type: ignore
import control.mppiisaac.utils.config_store            # type: ignore

import io
import hydra
import torch
import zerorpc

from scheduler import Scheduler
from monitor import Monitor
from motion import Dingo

from environment import SimulateWorld


@hydra.main(version_base=None, config_path="config", config_name="config_dingo_push")
def client(config):

    # Create world
    world = SimulateWorld.create_world(config, "blockage")
    world.configure_world()

    # Set objective function and mode for the mobile robot
    goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=config["mppi"].device)
    mode = torch.tensor([0., 0.], device=config["mppi"].device)

    # Scheduler
    size = world.params["environment"]["size"]

    step = world.params["scheduler"]["step"]
    goal = world.params["scheduler"]["goal"]

    scheduler = Scheduler(size, step, goal)

    # Running variables
    is_allowed_to_run = True
    is_executing_task = False
    current_exec_task = 0

    # Plotters
    monitor = Monitor.create_robot_monitor(world.simulation)
    
    # Get the tasks for the robot to execute
    tasks = scheduler.tasks(world.simulation)

    while is_allowed_to_run:
        if tasks and not is_executing_task and current_exec_task < len(tasks):
            actor, [x, y] = tasks[current_exec_task]
            move_mode = 0. if actor == 0 else 1.

            goal = torch.tensor([x, y, 0., 0., 0., 0., 1.], device=config["mppi"].device)
            mode = torch.tensor([move_mode, actor], device=config["mppi"].device)

            world.update_objective(goal, mode)
            is_executing_task = True

        if scheduler.task_succeeded(world.simulation, tasks[current_exec_task]):
            print(f"Tasked succeeded, elapsed time: {world.get_elapsed_time()}")
            is_executing_task = False
            current_exec_task += 1
        
        if current_exec_task >= len(tasks):
            is_allowed_to_run = False
    
        world.run_simulation(is_executing_task)
        monitor.add_data(world.simulation)

    world.destroy_simulation()
    monitor.plotter()


def ros_main():
    hydra.initialize(config_path="../../config", version_base=None)
    client(hydra.compose("config_dingo_push"))
