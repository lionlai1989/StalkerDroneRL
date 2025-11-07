import os
from glob import glob

from setuptools import setup

PACKAGE_NAME = "sdrl_object_mover"
VERSION = "1.0.0"
DESCRIPTION = "Object Mover Package"
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
        (os.path.join("share", PACKAGE_NAME, "models"), glob("models/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "move_object = sdrl_object_mover.move_object:main",
        ],
    },
)
