#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
from math import pi


class Constants:

    @staticmethod
    def get_basic_parameters():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if os.path.isfile(os.path.join(dir_path, "basic_parameters.json")):
            with open(os.path.join(dir_path, "basic_parameters.json"), "r") as f:
                basic_parameters = json.load(f)

            return basic_parameters
