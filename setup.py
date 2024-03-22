# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup_args = generate_distutils_setup(
    packages=['control', 'environment', 'isaacgym', 'monitor', 'motion', 'scheduler', 'sockets'],
    package_dir={'': 'src'},
    scripts=['scripts/node_client', 'scripts/node_server', 'scripts/node_collect'],
)

setup(**setup_args)