#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
from datetime import datetime

import hdf5storage
from os.path import join as pjoin

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_PATH + "/MWTsolver")

from configs.constants import Constants
from configs.logger import Logger
from dataloader.electric_field.electric_field_generator import ElectricFieldGenerator

import numpy as np
from scipy import linalg
import pylops



from csoperator import CSoperator

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/mwt_solver/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


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

        self.groundtruth_rel_perm = image.relative_permittivities.flatten("F")

        image_domain = np.linspace(-self.images_parameters["max_diameter"], self.images_parameters["max_diameter"],
                                   self.images_parameters["no_of_pixels"])
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

        self.measured_electric_field = self.electric_field_generator.generate_total_electric_field(self.groundtruth_rel_perm, x_domain, y_domain, full_pixel=True)
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

        # Initialize Total Electric Field
        self.total_electric_field = None
        self.sparse_coeffs = None

        self.error_E = []
        self.error_rel_perm = []
        self.loss = []

        self.k1 = np.sqrt(self.solver_parameters["alpha"] / self.incident_electric_field.size)  # B1 in Matlab code
        self.k2 = np.sqrt((1.0 - self.solver_parameters["alpha"] ) / self.measured_electric_field.size)  # B2 in Matlab code

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

            n += 1

            elapsed = time.time() - t0
            print('time elapse per iteration:' + str(elapsed) + 's')

    def update_total_electric_field(self):
        mat1 = np.identity(self.images_parameters["no_of_pixels"] ** 2) - np.matmul(self.green_function_D,
                                                                                    np.diag(
                                                                                        self.complex_rel_perm))
        mat2 = np.matmul(self.green_function_S, np.diag(self.complex_rel_perm))
        phi = np.concatenate((self.k1 * mat1, self.k2 * mat2), axis=0)

        q = np.concatenate((self.k1 * self.incident_electric_field, self.k2 * self.measured_electric_field), axis=0)

        self.total_electric_field = np.linalg.lstsq(phi, q, rcond=None)[0]

        error_E = np.linalg.norm(self.groundtruth_total_electric_field - self.total_electric_field, 'fro')/np.linalg.norm(self.groundtruth_total_electric_field, 'fro')
        loss = self.solver_parameters["alpha"]  * self.loss1() + (1.0 - self.solver_parameters["alpha"] ) * self.loss2()

        return error_E, loss

    def update_relative_permittivities(self):
        mat_C = (self.total_electric_field - self.incident_electric_field).transpose()
        mat_B2 = self.measured_electric_field.transpose()
        b = np.concatenate((self.k1 * mat_C.flatten("F"), self.k2 * mat_B2.flatten("F")), axis=0)
        b = np.concatenate((-b.real, b.imag), axis=0)

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
        Aopn = CSoperator(self.green_function_D, self.green_function_S, self.total_electric_field, self.k1,
                                 self.k2, self.dictionary, coeff)
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

        error_rel_perm = np.linalg.norm(self.groundtruth_complex_rel_perm.flatten() - self.complex_rel_perm)/np.linalg.norm(self.groundtruth_complex_rel_perm.flatten())
        loss = self.solver_parameters["alpha"] * self.loss1() + (1.0 - self.solver_parameters["alpha"] ) * self.loss2()
        return error_rel_perm, loss

    def loss1(self):
        mat1 = np.identity(self.images_parameters["no_of_pixels"] ** 2) - np.matmul(self.green_function_D, np.diag(self.complex_rel_perm.flatten("F")))
        mat2 = np.matmul(mat1, self.total_electric_field) - self.incident_electric_field

        return (np.linalg.norm(mat2, 'fro')**2) / self.incident_electric_field.size

    def loss2(self):
        mat = np.matmul(np.matmul(self.green_function_S, np.diag(self.complex_rel_perm.flatten("F"))), self.total_electric_field) - self.measured_electric_field
        return (np.linalg.norm(mat, 'fro')**2) / self.measured_electric_field.size