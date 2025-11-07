from setuptools import setup

PACKAGE_NAME = "sdrl_geometric_controller"
VERSION = "1.0.0"
DESCRIPTION = "Geometric Controller Package"
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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
