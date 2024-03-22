
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


def torch_to_bytes(torch_tensor) :
    buff = io.BytesIO()
    torch.save(torch_tensor, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(buffer):
    buff = io.BytesIO(buffer)
    return torch.load(buff)


@hydra.main(version_base=None, config_path="config", config_name="config_dingo_push")
def client(config):

    # Create world
    world = SimulateWorld.create_world(config, "blockage")
    world.configure_world()

    # Connect to the planner
    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")

    additions = world.create_additions()
    planner.add_to_env(additions)

    # Set objective function and mode for the mobile robot
    goal = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=config["mppi"].device)
    mode = torch.tensor([0., 0.], device=config["mppi"].device)

    planner.update_objective_goal(torch_to_bytes(goal))
    planner.update_objective_mode(torch_to_bytes(mode))

    # Scheduler
    size = world.params["environment"]["size"]

    step = world.params["scheduler"]["step"]
    goal = world.params["scheduler"]["goal"]

    scheduler = Scheduler(size, step, goal)

    # Robot
    dingo_robot = Dingo()

    # Running variables
    is_allowed_to_run = True
    is_task_scheduled = False

    # tasks
    ready = True
    tasks = None
    stage = 0

    # Plotters
    monitor = Monitor.create_robot_monitor(world.simulation)
    
    while is_allowed_to_run:

        if not is_task_scheduled and world.simulation.gym.get_elapsed_time(world.simulation.sim) > 1.0:
            tasks = scheduler.tasks(world.simulation)
            print(f"tasks {tasks}")

            is_task_scheduled = True

        planner.reset_rollout_sim(torch_to_bytes(world.simulation.dof_state[0]),
                                  torch_to_bytes(world.simulation.root_state[0]),
                                  torch_to_bytes(world.simulation.rigid_body_state[0]))

        if tasks is not None:
            if ready and stage < len(tasks):
                actor, [x, y] = tasks[stage]
                move_mode = 0. if actor == 0 else 1.

                goal = torch.tensor([x, y, 0., 0., 0., 0., 1.], device=config["mppi"].device)
                mode = torch.tensor([move_mode, actor], device=config["mppi"].device)

                planner.update_objective_goal(torch_to_bytes(goal))
                planner.update_objective_mode(torch_to_bytes(mode))
    
                ready = False

            action = bytes_to_torch(planner.command())
            dingo_robot.move(*action.numpy())

            if torch.any(torch.isnan(action)):
                print("nan action")
                action = torch.zeros_like(action)

            world.simulation.set_dof_velocity_target_tensor(action)

            if stage < len(tasks) and scheduler.task_succeeded(world.simulation, tasks[stage]):
                print(f"Tasked succeeded, elapsed time: {world.simulation.gym.get_elapsed_time(world.simulation.sim)}")
                ready = True
                stage += 1
            
            if stage == len(tasks):
                break
        
        monitor.add_data(world.simulation)

        world.simulation.step()

    world.destroy_viewer()
    world.destroy_sim()
    
    monitor.plotter()

def ros_main():
    hydra.initialize(config_path="../../config", version_base=None)
    client(hydra.compose("config_dingo_push"))
