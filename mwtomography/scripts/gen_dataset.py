#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mwtomography.dataloader.image import ImageGenerator

if __name__ == "__main__":
    # Test images generator
    image_generator = ImageGenerator(no_of_images=250000, shape='circle')
    images = image_generator.generate_images(test=False, nshapes='random')  # 'random', no of shapes, 'fixed_pattern'





