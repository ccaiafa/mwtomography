#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mwtomography.configs.constants import Constants
from mwtomography.dataloader.shapes.circle import Circle
from mwtomography.dataloader.shape_generators.shape_generator import ShapeGenerator


class CircleGenerator(ShapeGenerator):

    def __init__(self):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.min_radius = images_parameters["min_radius"]
        self.max_radius = images_parameters["max_radius"]

    #def generate_shapes(self, no_of_shapes, image_i, test):
    def generate_shapes(self, no_of_shapes):
        circles = []
        for i in range(no_of_shapes):
            #if test:
            #    radius = self.min_radius + (self.max_radius - self.min_radius) * super().TEST_RANDOM_VALUES[image_i - 1][i]
            #    center_x, center_y, relative_permittivity = super().get_test_shape_parameters(radius, image_i, i)
            #else:
            #    radius = self.min_radius + (self.max_radius - self.min_radius) * np.random.uniform()
            #    center_x, center_y, relative_permittivity = super().get_shape_parameters(radius)
            radius = self.min_radius + (self.max_radius - self.min_radius) * np.random.uniform()
            center_x, center_y, relative_permittivity = super().get_shape_parameters(radius)
            circle = Circle(radius, center_x, center_y, relative_permittivity)
            circles.append(circle)

        return circles

    def generate_shapes_pattern(self):
        circles = []
        # circle 1
        radius = 0.35,
        center_x = 0.10,
        center_y = -0.42,
        relative_permittivity = 1.30
        circle = Circle(radius, center_x, center_y, relative_permittivity)
        circles.append(circle)

        # circle 2
        radius = 0.33,
        center_x = -0.09,
        center_y = -0.36,
        relative_permittivity = 1.13
        circle = Circle(radius, center_x, center_y, relative_permittivity)
        circles.append(circle)

        # circle 3
        radius = 0.40,
        center_x = -0.02,
        center_y = 0.36,
        relative_permittivity = 1.30
        circle = Circle(radius, center_x, center_y, relative_permittivity)
        circles.append(circle)

        return circles

    def get_shape_name(self):
        return 'circles'


