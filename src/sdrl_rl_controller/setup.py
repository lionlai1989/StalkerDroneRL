import os
from glob import glob

from setuptools import setup

PACKAGE_NAME = "sdrl_rl_controller"
VERSION = "1.0.0"
DESCRIPTION = "Gymnasium + Stable-Baselines3 RL controller"
AUTHOR = "Chih-An Lion Lai"
LICENSE = "Apache-2.0"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_sac = sdrl_rl_controller.train_sac:main",
            "train_imitation = sdrl_rl_controller.train_imitation:main",
            "evaluate_controllers = sdrl_rl_controller.evaluate_controllers:main",
        ],
    },
)
