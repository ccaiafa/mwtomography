#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Image:
    def generate_relative_permittivities(self, x_domain, y_domain, shapes):
        self.shapes = shapes
        self.relative_permittivities = np.ones(np.shape(y_domain))
        for shape in shapes:
            relative_permittivity = shape.get_relative_permittivity()
            pixel_belongs_to_shape = shape.check_if_pixels_belong_to_shape(x_domain, y_domain)
            self.relative_permittivities[pixel_belongs_to_shape] = relative_permittivity

    def set_relative_permittivities(self, relative_permittivities):
        self.relative_permittivities = relative_permittivities

    def get_relative_permittivities(self):
        return self.relative_permittivities
