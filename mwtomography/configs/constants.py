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
        with open(os.path.join(dir_path, "basic_parameters.json"), "r") as f:
            basic_parameters = json.load(f)
            basic_parameters["physics"]["impedance_of_free_space"] = 120 * pi  # eta_0

            #basic_parameters["physics"]["k_0"] = 2 * pi / basic_parameters["physics"]["wavelength"]
            #basic_parameters["physics"]["omega"] = basic_parameters["physics"]["k_0"] * basic_parameters["physics"]["speed_of_light"]
            
            #basic_parameters["images"]["step_size"] = 2 * basic_parameters["images"]["max_diameter"] / (basic_parameters["images"]["no_of_pixels"] - 1)
            #basic_parameters["images"]["cell_area"] = basic_parameters["images"]["step_size"]^2
            
            #basic_parameters["physics"]["gamma"] = basic_parameters["physics"]["omega"] * basic_parameters["physics"]["vacuum_permittivity"] * basic_parameters["images"]["cell_area"]

            return basic_parameters
