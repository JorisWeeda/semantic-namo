# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup_args = generate_distutils_setup(
    packages=['control', 'environment', 'isaacgym', 'monitor', 'motion', 'scheduler'],
    package_dir={'': 'src'},
    scripts=['scripts/collector', 'scripts/controller', 'scripts/environment', 'scripts/visualisation',
             'tests/benchmark_global_planner', 'tests/benchmark_passage_nodes'],
)

setup(**setup_args)