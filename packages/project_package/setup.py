from setuptools import setup, find_packages
import pathlib
import os

root = pathlib.Path(__file__).parent
os.chdir(str(root))

setup(
    name='project-package',
    version='0.0.0',
    packages=find_packages(),
    description='Package for demo project.'
)