# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup_args = generate_distutils_setup(
    packages=['control', 'isaacgym', 'monitor', 'motion', 'scheduler', 'sockets'],
    package_dir={'': 'src'},
    scripts=['scripts/main_client', 'scripts/main_server', 'scripts/collector'],
)

setup(**setup_args)