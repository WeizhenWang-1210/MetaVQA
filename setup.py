# Please don't change the order of following packages!
import os
import sys
from os import path

from setuptools import setup, find_namespace_packages  # This should be place at top!

ROOT_DIR = os.path.dirname(__file__)


def get_version():
    context = {}
    with open('./metadrive/version.py', 'r') as file:
        exec(file.read(), context)
    return context['VERSION']


VERSION = get_version()


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


assert sys.version_info.major == 3 and sys.version_info.minor >= 6 and sys.version_info.minor < 12, \
    "python version >= 3.6, <3.12 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "documentation", "documentation.*", "build.*"))
print("We will install the following packages: ", packages)

install_requires = [
    "requests",
    "gymnasium>=0.28, <0.29",
    "numpy>=1.21.6, <=1.24.2",
    "matplotlib",
    "pandas",
    "pygame",
    "tqdm",
    "yapf",
    "seaborn",
    "tqdm",
    "progressbar",
    # "panda3d==1.10.8",
    "panda3d==1.10.13",
    "panda3d-gltf==0.13",  # 0.14 will bring some problems
    "panda3d-simplepbr",
    "pillow",
    "pytest",
    "opencv-python",
    "lxml",
    "scipy",
    "psutil",
    "geopandas",
    "shapely",
    "filelock"
]

nuplan_requirement = [
    "bokeh==2.4",
    "hydra-core",
    "chardet",
    "pyarrow",
    "aiofiles",
    "retry",
    "boto3",
    "aioboto3"
]

# Or try:
#   pip install git+https://github.com/waymo-research/waymo-open-dataset.git
waymo_requirement = [
    "waymo-open-dataset-tf-2.11.0==1.5.0"
]

cuda_requirement = [
    "cuda-python==12.0.0",
    "PyOpenGL==3.1.6",
    "PyOpenGL-accelerate==3.1.6",
    "pyrr==0.10.3",
    "glfw",
]

gym_requirement = [
    "gym>=0.19.0, <=0.26.0"
]

ros_requirement = [
    "zmq"
]

setup(
    name="metadrive-simulator",
    python_requires='>=3.6, <3.12',  # do version check with assert
    version=VERSION,
    description="An open-ended driving simulator with infinite scenes",
    url="https://github.com/metadriverse/metadrive",
    author="MetaDrive Team",
    author_email="quanyili0057@gmail.com, pzh@cs.ucla.edu",
    packages=packages,
    install_requires=install_requires,
    extras_require={
        "cuda": cuda_requirement,
        "nuplan": nuplan_requirement,
        "waymo": waymo_requirement,
        "gym": gym_requirement,
        "ros": ros_requirement,
        "all": nuplan_requirement + cuda_requirement
    },
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)

"""
How to publish to pypi and Draft github Release?  Noted by Zhenghao and Quanyi in Dec 27, 2020.

0. Checkout a new branch from main called releases/x.y.z

1. Rename VERSION in metadrive/version.py to x.y.z

2. Revise the version in metadrive/assets/version.txt, and compress the folder: zip -r assets.zip assets

3. commit changes and push this branch to remote

4. Draft a release on github with new version number and upload assets.zip 

5. Remove old files and ext_modules from setup() to get a clean wheel for all platforms in py3-none-any.wheel
    rm -rf dist/ build/ documentation/build/ metadrive_simulator.egg-info/ docs/build/

6. Rename current version to X.Y.Z.rcA, where A is arbitrary value represent "release candidate A". 
   This is really important since pypi do not support renaming and re-uploading. 
   Rename version in metadrive/versions.py 

7. Get wheel
    python setup.py sdist bdist_wheel

8. Upload to test channel
    twine upload --repository testpypi dist/*

9. Test as next line. If failed, change the version name and repeat 1, 2, 3, 4, 5.
    pip install --index-url https://test.pypi.org/simple/ metadrive

10. Rename current version to X.Y.Z in setup.py, rerun 1, 3 steps.

11. Upload to production channel 
    twine upload dist/*

12. Upload the generated .whl file to release files

13. merge this branch into main

!!!!!!!!!!!!! NOTE: please make sure that unzip assets.zip will generate a folder called assets instead of files  

"""
