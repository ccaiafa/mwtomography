#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import pywt

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mwtomography.dataloader.image.image_generator import ImageGenerator
from mwtomography.MWTsolver.mwt_solver import MWTsolver

from mwtomography.MWTsolver.mwt_solver_TV import MWTsolverTV
from matplotlib import pyplot as plt
import numpy as np
from mwtomography.utils.file_manager import FileManager


def plot_results(solver, path):
    N = solver.images_parameters['no_of_pixels']
    plt.figure(figsize=(5, 5), layout='constrained')
    plt.subplot(211)  # Error vs iter
    plt.plot(solver.error_E, label='error Total Electric Field')
    plt.plot(solver.error_rel_perm, label='error Complex Relative Permittivity')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title("Error vs iteration")
    plt.legend()

    plt.subplot(212)  # Loss vs iter
    plt.plot(solver.loss, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Loss vs iteration")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(5, 5), layout='constrained',
                            subplot_kw={'xticks': [], 'yticks': []})
    plt.subplot(231), plt.title("Ground Truth")  # Ground Truth Rel Perm
    plt.imshow(np.abs(solver.groundtruth_complex_rel_perm.reshape(N, N)))
    plt.subplot(232), plt.title("CS estimate")  # Estimated Rel Perm
    plt.imshow(np.abs(solver.complex_rel_perm.reshape(N, N)))
    plt.subplot(233), plt.title("Abs error:" + "{:.4f}".format(solver.error_rel_perm[-1]))  # Rel Perm Error
    plt.imshow(np.abs(solver.complex_rel_perm.reshape(N, N) - solver.groundtruth_complex_rel_perm.reshape(N, N)))

    plt.subplot(234), plt.title("Tot Elec Field")  # Ground Truth Total Electric Field
    plt.imshow(np.abs(solver.groundtruth_total_electric_field[:, 1].reshape(N, N)))
    plt.subplot(235), plt.title("Estimation")  # Estimated Rel Perm
    plt.imshow(np.abs(solver.total_electric_field[:, 1].reshape(N, N)))
    plt.subplot(236), plt.title("Abs error" + "{:.4f}".format(solver.error_E[-1]))  # Rel Perm Error
    plt.imshow(np.abs(
        solver.groundtruth_total_electric_field[:, 1].reshape(N, N) - solver.total_electric_field[:, 1].reshape(N,
                                                                                                                N)))
    plt.show()

    plt.savefig(path)
    plt.close("all")


def sparsify(image, p=0.01):
    print("image was sparsified")
    x = image.relative_permittivities
    # compute the 2D DWT
    type = "haar"
    coeffs = pywt.wavedec2(x, type, mode='periodization', level=None)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # keep largest coefficients
    nz_coeffs = round(p * 64 * 64)
    sorted_arr = np.sort(np.abs(arr.ravel()))
    sorted_arr = sorted_arr[::-1]
    threshold = sorted_arr[nz_coeffs - 1]

    arr_new = np.copy(arr)
    arr_new[np.abs(arr_new) < threshold] = 0

    coeffs_new = pywt.array_to_coeffs(arr_new, coeff_slices, output_format='wavedec2')

    # reconstruction
    x_rec = pywt.waverec2(coeffs_new, type, mode='periodization')

    image.relative_permittivities = x_rec

    return image


if __name__ == "__main__":
    # Test images generator
    image_generator = ImageGenerator(no_of_images=1, shape='circle')
    images = image_generator.generate_images(test=True,
                                             nshapes=3)  # 'random', no of shapes, 'fixed_pattern'

    #A = np.stack((images[0].relative_permittivities, images[1].relative_permittivities))



    # Apply CS reconstruction algorithm
    image = images[0]
    #image = sparsify(image, 0.05)
    # Load a precomputed dictionary
    #dictionary_type = 'wavelet_db2'
    #dictionary_type = 'overlap_patch'
    # dictionary_type = 'patch'
    dictionary_type = 'full'
    # dictionary_type = 'kronecker'
    if dictionary_type == 'ODL':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/ODL/1024x4096.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == 'dct':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/dct/1024x1024.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == "full":
        dictionary_file = ROOT_PATH + "/dictionary/trained_dict_32x32_epoch_1.pkl"
        # dictionary_file = ROOT_PATH + "/data/trainer/dictionary/sklearn/trained_dict_epoch_1.pkl"
        dict_trainer = FileManager.load(dictionary_file)
        D = dict_trainer.components_.transpose()
        D = torch.from_numpy(D)
    elif dictionary_type == "patch":
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/patch/trained_dict_epoch__patch_64x64_epoch_4_batch_size_125000_ncomps_1024_dict.pkl"
        Dpatch = FileManager.load(dictionary_file)
        plot_dict(Dpatch)
        #D = np.random.rand(64*64, 4*64*64)
        D = construct_full_dict(Dpatch)
    elif dictionary_type == "overlap_patch":
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/patch/trained_dict_epoch__patch_64x64_epoch_4_batch_size_125000_ncomps_1024_dict.pkl"
        Dpatch = FileManager.load(dictionary_file)
        #plot_dict(Dpatch)
        D = construct_full_dict_overlap(Dpatch)
    elif dictionary_type == "kronecker":
        # dictionary_file = ROOT_PATH + "/data/trainer/dictionary/kronecker/trained_dict_epoch_4_kron.pkl"
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/kronecker/trained_dict_epoch_9_kron_lambda05.pkl"
        aux = FileManager.load(dictionary_file)
        D = np.kron(aux[0], aux[1])
    elif dictionary_type == "wavelet_haar":
        dictionary_file = ROOT_PATH + "/dictionary/wavelet_haar_dict.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == "wavelet_db2":
        dictionary_file = ROOT_PATH + "/dictionary/wavelet_db2_dict.pkl"
        D = FileManager.load(dictionary_file)

    #plot_dict(D)

    # Test
    #x = image.relative_permittivities
    #x1 = image2vectorized_patches(x)
    #xt = vectorized_patches2image(x1)

    # Solve using sparse representation first
    solverCS = MWTsolver(image, D, ROOT_PATH)
    solverCS.inverse_problem_solver()
    #file_name = ROOT_PATH + "/data/reconstruction/CS_64x64.png"
    #plot_results(solverCS, file_name)

    # Refine solution using Total VAriation
    solver = MWTsolverTV(image, solverCS.complex_rel_perm)
    solver.inverse_problem_solver()

    #file_name = ROOT_PATH + "/data/reconstruction/CSTV_64x64.png"
    #plot_results(solver, file_name)
