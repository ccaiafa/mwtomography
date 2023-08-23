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


def image2vectorized_patches(x):
    # Input is [64, 64]
    P = x.shape[0]  # 64
    J = 16  # patch size = 16
    n_blocks = P // J  # n_blocs = 4
    x = x.reshape(P, n_blocks, J)  # [64, 4, 16]
    x = x.transpose(1, 2, 0)  # [4, 16, 64]
    x = x.reshape(n_blocks, J, n_blocks, J)  # [4, 16, 4, 16]
    x = x.transpose(3, 1, 2, 0)  # [16, 16, 4, 4]
    x = x.reshape(J, J, n_blocks * n_blocks)  # [16, 16, 4*4]
    x = x.transpose(2, 0, 1)  # [4*4, 16,16]
    x = x.reshape(n_blocks * n_blocks, J * J)  # [4*4, 16*16]
    x = x.transpose()  # [16*16, 4*4]
    x = x.flatten("F")  # [16*16*4*4,]
    return x

def vectorized_patches2image(x, P=64, J=16):
    # Input is [16*16*4*4,]
    # P = 64 number of rows and columns
    # J = 16 patch size
    n_blocks = P // J  # n_blocs = 4

    x = x.reshape(J * J, n_blocks * n_blocks, order="F")  # [16*16, 4*4]
    x = x.transpose()  # [4*4, 16*16]
    x = x.reshape(n_blocks * n_blocks, J, J)  # [4*4, 16, 16]
    x = x.transpose(1, 2, 0)  # [16, 16, 4*4]
    x = x.reshape(J, J, n_blocks, n_blocks)  # [16, 16, 4, 4]
    x = x.transpose(3, 1, 2, 0)  # [4, 16, 4, 16]
    x = x.reshape(n_blocks, J, P)  # [4, 16, 64]
    x = x.transpose(2, 0, 1)  # [64, 4, 16]
    x = x.reshape(P, P)  # [4, 16, 64]
    return x

def construct_full_dict(Dp, P=64, J=16):
    n_blocks = P // J  # n_blocs = 4
    D = np.kron(np.eye(n_blocks * n_blocks, dtype=int), Dp)  # [256*16, 16384]
    D = D.transpose()  # [16384, 256*16]
    D = D.reshape(-1, J*J, n_blocks * n_blocks, order="F")  # [16384, 16*16, 4*4]
    D = D.transpose(0, 2, 1)  # [16384, 4*4, 16*16]
    D = D.reshape(-1, n_blocks * n_blocks, J, J)  # [16384, 4*4, 16, 16]
    D = D.transpose(0, 2, 3, 1)  # [16384, 16, 16, 4*4]
    D = D.reshape(-1, J, J, n_blocks, n_blocks)  # [16384, 16, 16, 4, 4]
    D = D.transpose(0, 4, 2, 3, 1)  # [16384, 4, 16, 4, 16]
    D = D.reshape(-1, n_blocks, J, P)  # [16384, 4, 16, 64]
    D = D.transpose(0, 3, 1, 2)  # [16384, 64, 4, 16]
    D = D.reshape(-1, P, P)  # [16384, 64, 64]
    D = D.reshape(-1, P*P, order="F") # [16384, 64*64]
    D = D.transpose()

    #D =  D / np.sqrt(np.sum(D * D, axis=0))

    return D

def plot_dict(D):
    plt.figure(figsize=(10, 10))
    M = D.shape[0] # number of pixels
    N = D.shape[1] # number of atoms
    patch_size = (int(np.sqrt(M)), int(np.sqrt(M)))
    ncols = int(np.sqrt(N))
    for i in range(N):
        plt.subplot(ncols, ncols
                    , i + 1)
        plt.imshow(D[:, i].reshape(patch_size), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(
        "Dictionary learned from face patches\n",
        fontsize=16,)
    plt.show()


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


def compute_D(Dpatch):

    return #D


if __name__ == "__main__":
    # Test images generator
    image_generator = ImageGenerator(no_of_images=1, shape='circle')
    images = image_generator.generate_images(test=True,
                                             nshapes='fixed_pattern')  # 'random', no of shapes, 'fixed_pattern'

    # Apply CS reconstruction algorithm
    image = images[0]
    # Load a precomputed dictionary
    dictionary_type = 'patch'
    # dictionary_type = 'full'
    # dictionary_type = 'kronecker'
    if dictionary_type == 'ODL':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/ODL/1024x4096.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == 'dct':
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/dct/1024x1024.pkl"
        D = FileManager.load(dictionary_file)
    elif dictionary_type == "full":
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/sklearn/trained_dict_64x64_epoch_1.pkl"
        # dictionary_file = ROOT_PATH + "/data/trainer/dictionary/sklearn/trained_dict_epoch_1.pkl"
        dict_trainer = FileManager.load(dictionary_file)
        D = dict_trainer.components_.transpose()
    elif dictionary_type == "patch":
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/patch/trained_dict_epoch__patch_64x64_epoch_4_batch_size_125000_ncomps_1024_dict.pkl"
        Dpatch = FileManager.load(dictionary_file)
        plot_dict(Dpatch)
        #D = np.random.rand(64*64, 4*64*64)
        D = construct_full_dict(Dpatch)
    elif dictionary_type == "kronecker":
        # dictionary_file = ROOT_PATH + "/data/trainer/dictionary/kronecker/trained_dict_epoch_4_kron.pkl"
        dictionary_file = ROOT_PATH + "/data/trainer/dictionary/kronecker/trained_dict_epoch_9_kron_lambda05.pkl"
        aux = FileManager.load(dictionary_file)
        D = np.kron(aux[0], aux[1])

    # Test
    #x = image.relative_permittivities
    #x1 = image2vectorized_patches(x)
    #xt = vectorized_patches2image(x1)

    # Solve using sparse representation first
    solverCS = MWTsolver(image, D)
    solverCS.inverse_problem_solver()
    file_name = ROOT_PATH + "/data/reconstruction/CS_64x64.png"
    plot_results(solverCS, file_name)

    # Refine solution using Total VAriation
    #solver = MWTsolverTV(image, solverCS.complex_rel_perm)
    #solver.inverse_problem_solver()

    #file_name = ROOT_PATH + "/data/reconstruction/CSTV_64x64.png"
    #plot_results(solver, file_name)
