#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pylops import LinearOperator
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Kron_operator(LinearOperator):
    r"""Kronecker dictionary operator

    Applies ...

    Parameters
    ----------
    D1, D2 : :obj:`numpy.ndarray`
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    """

    def __init__(self, D1, D2, dtype=None):
        self.D1 = D1
        self.D2 = D2
        self.I1 = D1.shape[0]
        self.I2 = D2.shape[0]
        self.J1 = D1.shape[1]
        self.J2 = D2.shape[1]
        self.no_of_pixels = D1.shape[0] * D2.shape[0]
        self.Nat = D1.shape[1] * D2.shape[1]

        super().__init__(dtype=np.dtype(dtype), shape=(self.no_of_pixels, self.Nat))

    def _matvec(self, x):
        x = np.matmul(self.D1, np.reshape(x, [self.I1, self.I2]))
        return np.matmul(x, np.transpose(self.D2)).reshape(-1)

    def _rmatvec(self, y):
        y = np.matmul(np.transpose(self.D1), np.reshape(y, [self.J1, self.J2]))
        return np.matmul(y, self.D2).reshape(-1)
