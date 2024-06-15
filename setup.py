# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup_args = generate_distutils_setup(
    packages=['control', 'environment', 'isaacgym', 'monitor', 'motion', 'scheduler'],
    package_dir={'': 'src'},
    scripts=['scripts/collector', 'scripts/controller', 'scripts/environment', 'scripts/visualisation',
             'tests/experiment_0', 'tests/experiment_1', 'tests/experiment_2', 'tests/experiment_3',
             'tests/helper_plotter'],
)

setup(**setup_args)