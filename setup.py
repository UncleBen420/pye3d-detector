#!/usr/bin/env python

import os
import platform

from setuptools import find_packages
from skbuild import setup


if __name__ == "__main__":
    setup(
        packages=find_packages(),
        package_data={"pye3d": ["refraction_models/*.msgpack"]},
    )
