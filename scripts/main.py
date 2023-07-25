#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import hdf5storage
import sys
from os.path import join as pjoin

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.image.image_generator import ImageGenerator
from MWTsolver.mwt_solver import MWTsolver

from MWTsolver.mwt_solver_TV import MWTsolverTV
from matplotlib import pyplot as plt
import numpy as np
from utils.file_manager import FileManager



def plot_results(solver):
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
    plt.subplot(233), plt.title("Abs error:"+"{:.4f}".format(solver.error_rel_perm[-1]))  # Rel Perm Error
    plt.imshow(np.abs(solver.complex_rel_perm.reshape(N, N) - solver.groundtruth_complex_rel_perm.reshape(N, N)))

    plt.subplot(234), plt.title("Tot Elec Field")  # Ground Truth Total Electric Field
    plt.imshow(np.abs(solver.groundtruth_total_electric_field[:, 1].reshape(N, N)))
    plt.subplot(235), plt.title("Estimation")  # Estimated Rel Perm
    plt.imshow(np.abs(solver.total_electric_field[:, 1].reshape(N, N)))
    plt.subplot(236), plt.title("Abs error"+"{:.4f}".format(solver.error_E[-1]))  # Rel Perm Error
    plt.imshow(np.abs(
        solver.groundtruth_total_electric_field[:, 1].reshape(N, N) - solver.total_electric_field[:, 1].reshape(N,
                                                                                                                  N)))
    plt.show()


if __name__ == "__main__":
    # Test images generator
    image_generator = ImageGenerator(no_of_images=1, shape='circle')
    images = image_generator.generate_images(test=True, nshapes='fixed_pattern')  # 'random', no of shapes, 'fixed_pattern'

    # Apply CS reconstruction algorithm
    image = images[0]
    # Load a precomputed dictionary
    dictionary_type = 'sklearn'
    if dictionary_type == 'ODL':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/ODL/1024x4096.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == 'dct':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/dct/1024x1024.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == "sklearn":
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/sklearn/trained_dict_64x64_epoch_1.pkl"
        dict_trainer = FileManager.load(dictionary_file)
        D = dict_trainer.components_.transpose()

    # Solve using sparse representation first
    solverCS = MWTsolver(image, D)
    solverCS.inverse_problem_solver()

    plot_results(solverCS)

    # Refine solution using Total VAriation
    solver = MWTsolverTV(image, solverCS.complex_rel_perm)
    solver.inverse_problem_solver()

    plot_results(solver)




