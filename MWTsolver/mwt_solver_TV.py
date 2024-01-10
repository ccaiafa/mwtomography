#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
from datetime import datetime

from configs.constants import Constants
from configs import Logger
from dataloader.electric_field.electric_field_generator import ElectricFieldGenerator

import numpy as np
from scipy import linalg
import pylops
from matplotlib import pyplot as plt


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_PATH + "/MWTsolver")
from tvoperator import TVoperator

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/mwt_solver/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)

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
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(path, dpi=100)
    plt.close("all")

class MWTsolverTV:

    def __init__(self, image, init_guess=[]):
        self.total_electric_field = None
        LOG.info("Starting MWT solver")
        self.basic_parameters = Constants.get_basic_parameters()
        self.images_parameters = self.basic_parameters["images"]
        self.physics_parameters = self.basic_parameters['physics']
        self.solver_parameters = self.basic_parameters["TV_optimizer"]

        self.electric_field_generator = ElectricFieldGenerator()

        self.groundtruth_rel_perm = image.relative_permittivities.flatten("F")

        image_domain = np.linspace(-self.images_parameters["max_diameter"], self.images_parameters["max_diameter"],
                                   self.images_parameters["no_of_pixels"])
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

        self.measured_electric_field = self.electric_field_generator.generate_total_electric_field(
            self.groundtruth_rel_perm, x_domain, y_domain, full_pixel=True)
        self.incident_electric_field = self.electric_field_generator.generate_incident_electric_field(x_domain,
                                                                                                      y_domain)
        self.groundtruth_complex_rel_perm = -1j * self.electric_field_generator.angular_frequency * (
                self.groundtruth_rel_perm - 1.0) * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area

        x_domain = np.atleast_2d(x_domain.flatten("F")).T
        y_domain = np.atleast_2d(y_domain.flatten("F")).T

        self.groundtruth_total_electric_field = self.electric_field_generator.get_total_electric_field_transmitters(
            x_domain, y_domain,
            self.groundtruth_complex_rel_perm,
            self.incident_electric_field)

        self.green_function_D = self.electric_field_generator.green_function_D  # A1 in Matlab code
        self.green_function_S = self.electric_field_generator.green_function_S  # A2 in Matlab code

        # Initialize complex relative permittivities
        if self.solver_parameters["init"] == 'random':
            self.complex_rel_perm = -1j * np.random.rand(*self.groundtruth_rel_perm.shape) * 1e-5
        elif self.solver_parameters["init"] == 'ground_truth':
            self.complex_rel_perm = -1j * self.electric_field_generator.angular_frequency * (
                    self.groundtruth_rel_perm - 1) * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area
        elif self.solver_parameters["init"] == 'initial_guess':
            self.complex_rel_perm = init_guess


        # Initialize Total Electric Field
        self.total_electric_field = None
        self.sparse_coeffs = None

        self.error_E = []
        self.error_rel_perm = []
        self.loss = []

        self.k1 = np.sqrt(self.solver_parameters["alpha"] / self.incident_electric_field.size)  # B1 in Matlab code
        self.k2 = np.sqrt(
            (1.0 - self.solver_parameters["alpha"]) / self.measured_electric_field.size)  # B2 in Matlab code

    def inverse_problem_solver(self):
        n = 1
        loss_variation = np.infty
        loss_previous = np.infty

        while (n <= self.solver_parameters["max_iter"]) and (
                loss_variation > self.solver_parameters["threshold"]):
            t0 = time.time()
            LOG.info(f'''iter={n}/{self.solver_parameters["max_iter"]}''')

            error_E, loss = self.update_total_electric_field()
            LOG.info(f'''loss after updating Electric Field = {loss}, error_E={error_E}''')

            error_rel_perm, loss = self.update_relative_permittivities()
            LOG.info(f'''loss after updating Relative Permittivities = {loss}, error_rel_perm={error_rel_perm}''')

            self.error_E.append(error_E)
            self.error_rel_perm.append(error_rel_perm)
            self.loss.append(loss)

            loss_variation = np.abs(loss - loss_previous)
            loss_previous = loss

            elapsed = time.time() - t0
            print('time elapse per iteration:' + str(elapsed) + 's')
            file_name = ROOT_PATH + "/data/reconstruction/TV_64x64_iter_"+str(n)+".png"
            plot_results(self, file_name)

            n += 1

    def update_total_electric_field(self):
        mat1 = np.identity(self.images_parameters["no_of_pixels"] ** 2) - np.matmul(self.green_function_D,
                                                                                    np.diag(
                                                                                        self.complex_rel_perm))
        mat2 = np.matmul(self.green_function_S, np.diag(self.complex_rel_perm))
        phi = np.concatenate((self.k1 * mat1, self.k2 * mat2), axis=0)

        q = np.concatenate((self.k1 * self.incident_electric_field, self.k2 * self.measured_electric_field), axis=0)

        self.total_electric_field = np.linalg.lstsq(phi, q, rcond=None)[0]

        error_E = np.linalg.norm(self.groundtruth_total_electric_field - self.total_electric_field,
                                 'fro') / np.linalg.norm(self.groundtruth_total_electric_field, 'fro')
        loss = self.solver_parameters["alpha"] * self.loss1() + (1.0 - self.solver_parameters["alpha"]) * self.loss2()

        return error_E, loss

    def update_relative_permittivities(self):
        mat_C = (self.total_electric_field - self.incident_electric_field).transpose()
        mat_B2 = self.measured_electric_field.transpose()
        b = np.concatenate((self.k1 * mat_C.flatten("F"), self.k2 * mat_B2.flatten("F")), axis=0)
        b = np.concatenate((-b.real, b.imag), axis=0)

        mat_Q1 = linalg.khatri_rao(self.green_function_D, self.total_electric_field.transpose())
        mat_Q2 = linalg.khatri_rao(self.green_function_S, self.total_electric_field.transpose())
        mat_A = np.concatenate((self.k1 * mat_Q1, self.k2 * mat_Q2), axis=0)
        mat_A = np.concatenate((mat_A.imag, mat_A.real), axis=0)
        mat_PHI = -self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area * mat_A

        Aop = pylops.MatrixMult(mat_PHI)
        Aop.explicit = False  # temporary solution whilst PyLops gets updated

        coeff = -self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area
        Aopn = TVoperator(self.green_function_D, self.green_function_S, self.total_electric_field, self.k1,
                          self.k2, coeff)
        Aopn.explicit = False  # temporary solution whilst PyLops gets updated

        Aopn.compute_G()

        # Total Variation (TV) solver
        Dop = [pylops.FirstDerivative((32, 32), axis=0, edge=False, kind="backward", dtype=np.float128),
               pylops.FirstDerivative((32, 32), axis=1, edge=False, kind="backward", dtype=np.float128), ]

        xinv = pylops.optimization.sparsity.splitbregman(
            Aopn,
            b,
            Dop,
            niter_outer=self.solver_parameters["niter_out"],
            niter_inner=self.solver_parameters["niter_in"],
            mu=self.solver_parameters["mu"],
            epsRL1s=[self.solver_parameters["lambda"], self.solver_parameters["lambda"]],
            tol=self.solver_parameters["tol"],
            tau=self.solver_parameters["tau"],
            show=self.solver_parameters["show"],
            **dict(iter_lim=self.solver_parameters["iter_lim"], damp=self.solver_parameters["damp"])
        )[0]

        self.complex_rel_perm = -1j * self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area * xinv

        error_rel_perm = np.linalg.norm(
            self.groundtruth_complex_rel_perm.flatten() - self.complex_rel_perm) / np.linalg.norm(
            self.groundtruth_complex_rel_perm.flatten())
        loss = self.solver_parameters["alpha"] * self.loss1() + (1.0 - self.solver_parameters["alpha"]) * self.loss2()
        return error_rel_perm, loss

    def loss1(self):
        mat1 = np.identity(self.images_parameters["no_of_pixels"] ** 2) - np.matmul(self.green_function_D, np.diag(
            self.complex_rel_perm.flatten("F")))
        mat2 = np.matmul(mat1, self.total_electric_field) - self.incident_electric_field

        return (np.linalg.norm(mat2, 'fro') ** 2) / self.incident_electric_field.size

    def loss2(self):
        mat = np.matmul(np.matmul(self.green_function_S, np.diag(self.complex_rel_perm.flatten("F"))),
                        self.total_electric_field) - self.measured_electric_field
        return (np.linalg.norm(mat, 'fro') ** 2) / self.measured_electric_field.size
