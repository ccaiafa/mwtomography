#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.fftpack import dct, idct
import pywt
from configs.constants import Constants
from utils.file_manager import FileManager
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


if __name__ == "__main__":
    basic_parameters = Constants.get_basic_parameters()
    no_of_pixels = basic_parameters["images"]["no_of_pixels"]
    coeffs = np.zeros([no_of_pixels, no_of_pixels])

    # DCT dict
    D = np.zeros([no_of_pixels**2, no_of_pixels**2])
    for i in range(no_of_pixels):
        for j in range(no_of_pixels):
            coeffs[i, j] = 1.0
            atom = idct(coeffs)
            coeffs[i, j] = 0.0

            D[:, i * no_of_pixels + j] = atom.flatten()

    dictionary_file = ROOT_PATH + "/data/dictionaries/dct/1024x1024.pkl"
    FileManager.save(D, dictionary_file)







