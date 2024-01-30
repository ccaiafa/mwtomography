#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
from datetime import datetime
from matplotlib import pyplot as plt
import torch

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_PATH + "/MWTsolver")

from mwtomography.configs.constants import Constants
from mwtomography.configs.logger import Logger
from mwtomography.dataloader.electric_field.electric_field_generator import ElectricFieldGenerator

import numpy as np
import pylops


from mwtomography.MWTsolver.csoperator import CSoperator
#from csoperator import CSoperator

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
        solver.groundtruth_total_electric_field[:, 1].reshape(N, N) - solver.total_electric_field[:, 1].reshape(N, N)))

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(path, dpi=100)
    plt.close("all")


class MWTsolver:

    def __init__(self, image, D, init_guess=[]):
        self.dictionary = D
        self.total_electric_field = None
        LOG.info("Starting MWT solver")
        self.basic_parameters = Constants.get_basic_parameters()
        self.images_parameters = self.basic_parameters["images"]
        self.physics_parameters = self.basic_parameters['physics']
        self.solver_parameters = self.basic_parameters["CS_optimizer"]

        self.electric_field_generator = ElectricFieldGenerator()

        if torch.is_tensor(image.relative_permittivities):
            self.groundtruth_rel_perm = image.relative_permittivities.t().flatten()
        else:
            self.groundtruth_rel_perm = image.relative_permittivities.flatten("F")

        image_domain = np.linspace(-self.images_parameters["max_diameter"], self.images_parameters["max_diameter"],
                                   self.images_parameters["no_of_pixels"])
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

        self.measured_electric_field, __ = self.electric_field_generator.generate_total_electric_field(self.groundtruth_rel_perm, x_domain, y_domain, full_pixel=True)
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

        self.green_function_D = self.electric_field_generator.green_function_D # A1 in Matlab code
        self.green_function_S = self.electric_field_generator.green_function_S  # A2 in Matlab code

        # Initialize complex relative permittivities
        if self.solver_parameters["init"] == 'random':
            self.complex_rel_perm = -1j * np.random.rand(*self.groundtruth_rel_perm.shape) * 1e-5
        elif self.solver_parameters["init"] == 'ground_truth':
            self.complex_rel_perm = -1j * self.electric_field_generator.angular_frequency * (
                    self.groundtruth_rel_perm - 1) * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area
        elif self.solver_parameters["init"] == 'zero':
            self.complex_rel_perm = -1j * torch.zeros(*self.groundtruth_rel_perm.shape)
            #self.complex_rel_perm = -1j * np.zeros(*self.groundtruth_rel_perm.shape)

        # Initialize Total Electric Field
        self.total_electric_field = None
        self.sparse_coeffs = None

        self.error_E = []
        self.error_rel_perm = []
        self.loss = []

        self.k1 = np.sqrt(self.solver_parameters["alpha"] / self.incident_electric_field.numel())  # B1 in Matlab code
        self.k2 = np.sqrt((1.0 - self.solver_parameters["alpha"] ) / self.measured_electric_field.numel())  # B2 in Matlab code

    def inverse_problem_solver(self):
        n = 1
        loss_variation = np.infty
        loss_previous = np.infty

        while (n <= self.solver_parameters["main_loop"]["max_iter"]) and (loss_variation > self.solver_parameters["main_loop"]["threshold"]):
            t0 = time.time()
            LOG.info(f'''iter={n}/{self.solver_parameters["main_loop"]["max_iter"]}''')

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
            file_name = ROOT_PATH + "/data/reconstruction/CS_64x64_iter_"+str(n)+".png"
            plot_results(self, file_name)

            n += 1


    def update_total_electric_field(self):
        aux = torch.eye(self.images_parameters["no_of_pixels"] ** 2)
        mat1 = aux.type(torch.complex128) - torch.matmul(self.green_function_D, torch.diag(self.complex_rel_perm).type(torch.complex128))
        mat2 = torch.matmul(self.green_function_S, torch.diag(self.complex_rel_perm).type(torch.complex128))
        phi = torch.cat((self.k1 * mat1, self.k2 * mat2), axis=0)

        q = torch.cat((self.k1 * self.incident_electric_field, self.k2 * self.measured_electric_field), axis=0)

        #self.total_electric_field = np.linalg.lstsq(phi, q, rcond=None)[0]
        self.total_electric_field = torch.linalg.lstsq(mat1, self.incident_electric_field, rcond=None)[0]

        error_E = torch.linalg.norm(self.groundtruth_total_electric_field - self.total_electric_field, 'fro')/torch.linalg.norm(self.groundtruth_total_electric_field, 'fro')
        loss = self.solver_parameters["alpha"]  * self.loss1() + (1.0 - self.solver_parameters["alpha"] ) * self.loss2()

        return error_E, loss

    def update_relative_permittivities(self):
        mat_C = (self.total_electric_field - self.incident_electric_field).t()
        mat_B2 = self.measured_electric_field.t()
        #b = np.concatenate((self.k1 * mat_C.flatten("F"), self.k2 * mat_B2.flatten("F")), axis=0)

        b = mat_B2.t().flatten()
        #b = mat_B2.flatten("F")
        b = torch.cat((-b.real, b.imag), axis=0)

        #mat_Q1 = linalg.khatri_rao(self.green_function_D, self.total_electric_field.transpose())
        #mat_Q2 = linalg.khatri_rao(self.green_function_S, self.total_electric_field.transpose())
        #mat_A = np.concatenate((self.k1 * mat_Q1, self.k2 * mat_Q2), axis=0)
        #mat_A = np.concatenate((mat_A.imag, mat_A.real), axis=0)
        #mat_PHI = -self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area * np.matmul(mat_A, self.dictionary)
        #col_norms = linalg.norm(mat_PHI, axis=0)
        #mat_PHI = mat_PHI / col_norms

        #Aop = pylops.MatrixMult(mat_PHI)
        #Aop.explicit = False  # temporary solution whilst PyLops gets updated

        ######
        coeff = -self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area
        Aopn = CSoperator(self.green_function_D.numpy(), self.green_function_S.numpy(), self.total_electric_field.numpy(), self.k1,
                                 self.k2, self.dictionary.numpy(), coeff)
        Aopn.explicit = False  # temporary solution whilst PyLops gets updated

        # Precompute matrix G and norm of columns
        Aopn.norm2col_op()

        # correct independent term Ax - b = A(Ds + constant) - b = ADs - (b-bp)
        bp = self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area * Aopn.q_times_one()
        b = b - bp


        self.sparse_coeffs, niterf, costf = pylops.optimization.sparsity.fista(Aopn, b, x0=self.sparse_coeffs, niter=self.solver_parameters['sparse_solver']['max_iter'], eps=self.solver_parameters['sparse_solver']['lambda'],
                                               tol=self.solver_parameters['sparse_solver']['threshold'], show=self.solver_parameters['verbose'])

        self.sparse_coeffs = self.sparse_coeffs / Aopn.norm2col

        self.complex_rel_perm = -1j * self.electric_field_generator.angular_frequency * self.electric_field_generator.vacuum_permittivity * self.electric_field_generator.pixel_area * (np.matmul(self.dictionary, self.sparse_coeffs))

        error_rel_perm = torch.linalg.norm(torch.tensor(self.groundtruth_complex_rel_perm).flatten() - self.complex_rel_perm)/torch.linalg.norm(torch.tensor(self.groundtruth_complex_rel_perm).flatten())
        loss = self.solver_parameters["alpha"] * self.loss1() + (1.0 - self.solver_parameters["alpha"] ) * self.loss2()
        return error_rel_perm, loss

    def loss1(self):
        aux = torch.eye(self.images_parameters["no_of_pixels"] ** 2)
        mat1 = aux.type(torch.complex128) - torch.matmul(self.green_function_D.type(torch.complex128), torch.diag(self.complex_rel_perm.t().flatten()).type(torch.complex128))
        mat2 = torch.matmul(mat1, self.total_electric_field) - self.incident_electric_field

        return (torch.linalg.norm(mat2, 'fro')**2) / self.incident_electric_field.numel()

    def loss2(self):
        mat = torch.matmul(np.matmul(self.green_function_S, np.diag(self.complex_rel_perm.t().flatten().type(torch.complex128))), self.total_electric_field) - self.measured_electric_field
        return (torch.linalg.norm(mat, 'fro')**2) / self.measured_electric_field.numel()