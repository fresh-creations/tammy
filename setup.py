from pathlib import Path
from setuptools import find_packages, setup

project_dir = Path(__file__).parent

setup(
    name='tammy',
    version='0.1',
    url='https://github.com/jonasdoevenspeck/Tammy2.git',
    author='Jonas Doevenspeck',
    author_email='jonasdoevenspeck@gmail.com',
    description='Music-based AI video generation',
    packages=find_packages(include=["tammy"]),    
    install_requires=[project_dir.joinpath("requirements.txt").read_text().split("\n"),
        "clip @ git+https://github.com/openai/CLIP.git"]
    )
