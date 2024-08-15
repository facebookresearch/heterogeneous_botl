#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

root_dir = os.path.dirname(__file__)

with open(os.path.join(root_dir, "requirements.txt"), "r") as fh:
    install_requires = [
        line.strip() for line in fh.readlines() if not line.startswith("#")
    ]

setup(
    name="heterogeneous_botl",
    version="0.1",
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
)
