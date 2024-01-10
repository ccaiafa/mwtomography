#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from matplotlib import pyplot as plt

import numpy as np
import pywt
from mwtomography.utils.file_manager import FileManager

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mwtomography.dataloader.image import ImageGenerator


def plot_dict(D):
    plt.figure(figsize=(10, 10))
    M = D.shape[0] # number of pixels
    N = D.shape[1] # number of atoms
    for i in range(N):
        plt.subplot(int(np.sqrt(N)), int(np.sqrt(N))
                    , i + 1)
        plt.imshow(D[:, i].reshape(int(np.sqrt(M)), int(np.sqrt(M)), order="F"), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(
        "Wavelet Dictionary\n",
        fontsize=16,)
    plt.show()

if __name__ == "__main__":
    # Generate image
    image_generator = ImageGenerator(no_of_images=1, shape='circle')
    images = image_generator.generate_images(test=True,
                                             nshapes='fixed_pattern')  # 'random', no of shapes, 'fixed_pattern'
    image = images[0].relative_permittivities

    # compute the 2D DWT
    type = "db2"
    coeffs = pywt.wavedec2(image, type, mode='periodization', level=None)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # keep largest coefficients
    p = 0.075
    nz_coeffs = round(p * 64 * 64)
    sorted_arr = np.sort(np.abs(arr.ravel()))
    sorted_arr = sorted_arr[::-1]
    threshold = sorted_arr[nz_coeffs - 1]

    arr_new = np.copy(arr)
    arr_new[np.abs(arr_new) < threshold] = 0

    coeffs_new = pywt.array_to_coeffs(arr_new, coeff_slices, output_format='wavedec2')

    # reconstruction
    image_rec = pywt.waverec2(coeffs_new, type, mode='periodization')

    image_comparison = np.concatenate((image, image_rec, np.abs(image - image_rec)), axis=1)
    plt.imshow(image_comparison)
    plt.show()

    print("error = ", str(np.linalg.norm(image-image_rec)/np.linalg.norm(image)))
    print("")

    # Create Wavelet Dictionary
    X = np.eye(64 * 64)
    D = np.zeros([64 * 64, 64 * 64])
    for column in range(64 * 64):
        print("column=", str(column), "/", str(64 * 64))
        arr_test = X[:, column].reshape(64, 64, order="F")
        coeffs = pywt.array_to_coeffs(arr_test, coeff_slices, output_format='wavedec2')
        image_rec = pywt.waverec2(coeffs, type, mode='periodization')
        D[:, column] = image_rec.reshape(64 * 64, order="F")
    plot_dict(D)


    # Compare reconstructions
    image_rec2 = np.matmul(D, arr.reshape(64 * 64, order="F"))
    image_rec2 = image_rec2.reshape(64, 64, order="F")
    image_comparison = np.concatenate((image, image_rec2, np.abs(image - image_rec2)), axis=1)
    plt.imshow(image_comparison)
    plt.show()

    print("error = ", str(np.linalg.norm(image-image_rec2)/np.linalg.norm(image)))
    print("")

    # Save dictionary
    filename_dict = os.path.join(
        ROOT_PATH + "/dictionary/wavelet_" + type + "_dict.pkl")
    FileManager.save(D, filename_dict)










