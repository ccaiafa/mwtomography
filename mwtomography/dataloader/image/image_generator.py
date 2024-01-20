#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np

from configs.constants import Constants

from mwtomography.dataloader.electric_field.electric_field_generator import ElectricFieldGenerator
from mwtomography.dataloader.shape_generators.rectangle_generator import RectangleGenerator
from mwtomography.dataloader.shape_generators.circle_generator import CircleGenerator
from mwtomography.dataloader.image.image import Image

class ImageGenerator:

    def __init__(self, no_of_images, shape):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.no_of_images = no_of_images
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        if shape == 'rectangle':
            self.shape_generator = RectangleGenerator()
        elif shape == 'circle':
            self.shape_generator = CircleGenerator()
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_images(self, test, nshapes='random'):
        shape_name = self.shape_generator.get_shape_name()
        images = []

        for image_i in range(1, self.no_of_images + 1):
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

            if nshapes == 'random':
                no_of_shapes = int(np.ceil((3 * np.random.uniform()) + 1e-2))
                shapes = self.shape_generator.generate_shapes(no_of_shapes)
            elif type(nshapes) == int:
                no_of_shapes = nshapes
                shapes = self.shape_generator.generate_shapes(no_of_shapes)
            else:
                no_of_shapes = 3
                shapes = self.shape_generator.generate_shapes_pattern()

            image = Image()
            image.generate_relative_permittivities(x_domain, y_domain, shapes)
            measured_electric_field, total_electric_field = self.electric_field_generator.generate_electric_field(image, x_domain, y_domain, True)
            image.set_measured_electric_field(measured_electric_field)
            image.set_total_electric_field(total_electric_field)
            images.append(image)

        images = np.array(images)

        return images
