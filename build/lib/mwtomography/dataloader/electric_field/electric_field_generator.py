#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import pi
import torch

import numpy as np
from scipy.special import hankel1

from mwtomography.configs.constants import Constants
from mwtomography.dataloader.electric_field.electric_field import ElectricField
from mwtomography.utils.coordinates_converter import CoordinatesConverter


class ElectricFieldGenerator:

    def __init__(self, no_of_pixels=None, no_of_receivers=None, no_of_transmitters=None, max_diameter=None,
                 wavelength=None, receiver_radius=None, transmitter_radius=None, wave_type=None):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        images_parameters = basic_parameters["images"]
        speed_of_light = 3e8
        vacuum_permittivity = 8.85e-12
        self.impedance_of_free_space = 376.99
        if no_of_pixels is None:
            self.no_of_pixels = images_parameters["no_of_pixels"]
        else:
            self.no_of_pixels = no_of_pixels
        if no_of_receivers is None:
            self.no_of_receivers = physics_parameters["no_of_receivers"]
        else:
            self.no_of_receivers = no_of_receivers
        if no_of_transmitters is None:
            self.no_of_transmitters = physics_parameters["no_of_transmitters"]
        else:
            self.no_of_transmitters = no_of_transmitters
        if max_diameter is None:
            self.max_diameter = images_parameters["max_diameter"]
        else:
            self.max_diameter = max_diameter
        if wavelength is None:
            self.wave_number = 2 * np.pi / physics_parameters["wavelength"]
        else:
            self.wave_number = 2 * np.pi / wavelength
        self.angular_frequency = self.wave_number * speed_of_light
        self.vacuum_permittivity = vacuum_permittivity
        self.pixel_length = 2 * self.max_diameter / (self.no_of_pixels - 1)
        self.pixel_area = self.pixel_length ** 2
        if receiver_radius is None:
            self.receiver_radius = physics_parameters["receiver_radius"]
        else:
            self.receiver_radius = receiver_radius
        if transmitter_radius is None:
            self.transmitter_radius = physics_parameters["transmitter_radius"]
        else:
            self.transmitter_radius = transmitter_radius

        self.wave_incidence = physics_parameters["wave_incidence"]

        if wave_type is None:
            self.wave_type = physics_parameters["wave_type"]
        else:
            self.wave_type = wave_type
        self.electric_field_coefficient = 1j * self.wave_number * self.impedance_of_free_space
        self.equivalent_radius = np.sqrt(self.pixel_area / pi)
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        self.x_domain, self.y_domain = np.meshgrid(image_domain, -image_domain)
        self.green_function_S = self.compute_GS(self.x_domain, self.y_domain)
        self.green_function_D = self.compute_GD(self.x_domain, self.y_domain)

    def generate_electric_field(self, image, x_domain, y_domain, full_pixel=False):
        relative_permittivities = torch.tensor(image.get_relative_permittivities())
        measured_electric_field, total_electric_field = self.generate_total_electric_field(relative_permittivities,
                                                                                           x_domain, y_domain,
                                                                                           full_pixel)
        return ElectricField(measured_electric_field), ElectricField(total_electric_field)

    def generate_total_electric_field(self, relative_permittivities, x_domain, y_domain, full_pixel=False):
        complex_relative_permittivities = \
            -1j * self.angular_frequency * (relative_permittivities - 1) * self.vacuum_permittivity * self.pixel_area
        pixels_with_circle = relative_permittivities != 1
        if not (full_pixel):
            x_domain = x_domain[pixels_with_circle]
            y_domain = y_domain[pixels_with_circle]
        x_domain = np.atleast_2d(x_domain.flatten("F")).T
        y_domain = np.atleast_2d(y_domain.flatten("F")).T
        if not (full_pixel):
            complex_relative_permittivities = complex_relative_permittivities[pixels_with_circle]
        if torch.is_tensor(complex_relative_permittivities):
            complex_relative_permittivities = torch.flatten(torch.transpose(complex_relative_permittivities, -2, -1))
        else:
            complex_relative_permittivities = complex_relative_permittivities.flatten("F")

        x_receivers, y_receivers, _ = self.get_antennas_coordinates(self.no_of_receivers, self.receiver_radius)
        incident_electric_field = self.generate_incident_electric_field(x_domain, y_domain)

        total_electric_field_transmitters = self.get_total_electric_field_transmitters(x_domain, y_domain,
                                                                                       complex_relative_permittivities,
                                                                                       incident_electric_field)

        # x_circles, x_receivers = np.meshgrid(x_domain, x_receivers)
        # y_circles, y_receivers = np.meshgrid(y_domain, y_receivers)
        # dist_receivers_circles = np.sqrt((x_circles - x_receivers) ** 2 + (y_circles - y_receivers) ** 2)
        # integral_receivers = torch.from_numpy( \
        #    self.electric_field_coefficient * (1j / 4) * hankel1(0, self.wave_number * dist_receivers_circles)
        # )
        # self.green_function_S = integral_receivers

        if not torch.is_tensor(complex_relative_permittivities):
            complex_relative_permittivities = torch.from_numpy(complex_relative_permittivities)

        total_electric_field = torch.matmul(
            torch.matmul(self.green_function_S, torch.diag(complex_relative_permittivities)),
            total_electric_field_transmitters)

        # total_electric_field = np.matmul(np.matmul(integral_receivers, np.diag(complex_relative_permittivities)),
        #                                 total_electric_field_transmitters)
        return total_electric_field, total_electric_field_transmitters  # Es, Et

    def compute_GS(self, x_domain, y_domain):
        x_domain = np.atleast_2d(x_domain.flatten("F")).T
        y_domain = np.atleast_2d(y_domain.flatten("F")).T

        x_receivers, y_receivers, _ = self.get_antennas_coordinates(self.no_of_receivers, self.receiver_radius)

        x_circles, x_receivers = np.meshgrid(x_domain, x_receivers)
        y_circles, y_receivers = np.meshgrid(y_domain, y_receivers)
        dist_receivers_circles = np.sqrt((x_circles - x_receivers) ** 2 + (y_circles - y_receivers) ** 2)

        GS = torch.tensor(
            self.electric_field_coefficient * (1j / 4) * hankel1(0, self.wave_number * dist_receivers_circles))

        return GS

    def compute_GD(self, x_domain, y_domain):
        x_domain = np.atleast_2d(x_domain.flatten("F")).T
        y_domain = np.atleast_2d(y_domain.flatten("F")).T
        no_of_pixels_with_circle = max(np.shape(x_domain))
        x_domain_with_circles, x_domain_with_circles_2 = np.meshgrid(x_domain, x_domain)
        y_domain_with_circles, y_domain_with_circles_2 = np.meshgrid(y_domain, y_domain)
        dist_between_pixels_with_circles = np.sqrt((x_domain_with_circles - x_domain_with_circles_2) ** 2 +
                                                   (y_domain_with_circles - y_domain_with_circles_2) ** 2)
        dist_between_pixels_with_circles = dist_between_pixels_with_circles + np.identity(no_of_pixels_with_circle)

        integral_1 = 1j / 4 * hankel1(0, self.wave_number * dist_between_pixels_with_circles)
        phi = self.electric_field_coefficient * integral_1
        phi = phi * (np.ones(no_of_pixels_with_circle) - np.identity(no_of_pixels_with_circle))

        integral_2 = 1j / 4 * (2 / (self.wave_number * self.equivalent_radius) *
                               hankel1(1, self.wave_number * self.equivalent_radius) +
                               4 * 1j / ((self.wave_number ** 2) * self.pixel_area))
        phi = torch.tensor(phi + self.electric_field_coefficient * integral_2 * np.identity(no_of_pixels_with_circle))

        return phi

    def generate_incident_electric_field(self, x_domain, y_domain):
        x_transmitters, y_transmitters, transmitter_angles = \
            self.get_antennas_coordinates(self.no_of_transmitters, self.transmitter_radius)
        if self.wave_type == self.wave_incidence["plane_wave"]:
            wave_number_x = self.wave_number * np.cos(transmitter_angles)
            wave_number_y = self.wave_number * np.sin(transmitter_angles)
            incident_electric_field = np.exp(
                np.matmul(1j * x_domain, wave_number_x).T + np.matmul(1j * y_domain, wave_number_y).T)
        else:
            circle_x, transmitter_x = np.meshgrid(x_domain.T, x_transmitters.T)
            circle_y, transmitter_y = np.meshgrid(y_domain.T, y_transmitters.T)
            dist_transmitter_circles = np.sqrt((circle_x - transmitter_x) ** 2 + (circle_y - transmitter_y) ** 2)
            transposed_electric_field = \
                1j * self.wave_number * self.impedance_of_free_space * 1j / 4 * \
                hankel1(0, self.wave_number * dist_transmitter_circles)
            incident_electric_field = transposed_electric_field.T

        return torch.tensor(incident_electric_field)

    @staticmethod
    def get_antennas_coordinates(no_of_antennas, antenna_radius):
        antenna_angles_polar = np.linspace(0, 2 * np.pi, no_of_antennas + 1)
        antenna_angles_polar = antenna_angles_polar[:-1]
        antenna_angles_polar = np.atleast_2d(antenna_angles_polar.flatten("F")).T
        antenna_angles, antenna_radii = np.meshgrid(antenna_angles_polar, antenna_radius)
        antenna_angles = np.atleast_2d(antenna_angles.flatten("F")).T
        antenna_radii = np.atleast_2d(antenna_radii.flatten("F")).T
        x_antennas, y_antennas = CoordinatesConverter.pol2cart(antenna_radii, antenna_angles)
        return x_antennas, y_antennas, antenna_angles_polar

    def get_total_electric_field_transmitters(self, x_domain, y_domain, complex_relative_permittivities,
                                              incident_electric_field):
        no_of_pixels = max(np.shape(x_domain))
        # x_domain_with_circles, x_domain_with_circles_2 = np.meshgrid(x_domain, x_domain)
        # y_domain_with_circles, y_domain_with_circles_2 = np.meshgrid(y_domain, y_domain)
        # dist_between_pixels_with_circles = np.sqrt((x_domain_with_circles - x_domain_with_circles_2) ** 2 +
        #                                           (y_domain_with_circles - y_domain_with_circles_2) ** 2)
        # dist_between_pixels_with_circles = dist_between_pixels_with_circles + np.identity(no_of_pixels_with_circle)

        # integral_1 = 1j / 4 * hankel1(0, self.wave_number * dist_between_pixels_with_circles)
        # phi = self.electric_field_coefficient * integral_1
        # phi = phi * (np.ones(no_of_pixels_with_circle) - np.identity(no_of_pixels_with_circle))

        # integral_2 = 1j / 4 * (2 / (self.wave_number * self.equivalent_radius) *
        #                       hankel1(1, self.wave_number * self.equivalent_radius) +
        #                       4 * 1j / ((self.wave_number ** 2) * self.pixel_area))
        # phi = torch.tensor(phi + self.electric_field_coefficient * integral_2 * np.identity(no_of_pixels_with_circle))

        # self.green_function_D = phi

        # total_electric_field_transmitters = \
        #    (torch.linalg.solve(np.identity(no_of_pixels_with_circle) - np.matmul(phi, np.diag(complex_relative_permittivities))),
        #        incident_electric_field)

        if not torch.is_tensor(complex_relative_permittivities):
            complex_relative_permittivities = torch.from_numpy(complex_relative_permittivities)
        total_electric_field_transmitters = \
            torch.linalg.solve(
                (torch.eye(no_of_pixels) - torch.matmul(self.green_function_D,
                                                        torch.diag(complex_relative_permittivities))),
                incident_electric_field)

        return total_electric_field_transmitters
