import os
from glob import glob

from setuptools import setup

PACKAGE_NAME = "sdrl_bringup"
VERSION = "1.0.0"
DESCRIPTION = "Bringup the whole Stalker Drone simulation environment"
AUTHOR = "Chih-An Lion Lai"
LICENSE = "Apache-2.0"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*launch.[pxy][yma]*")),
        (os.path.join("share", PACKAGE_NAME, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", PACKAGE_NAME, "worlds"), glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
