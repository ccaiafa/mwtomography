#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pylops import LinearOperator
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class TVoperator(LinearOperator):
    r"""CS Tomography operator

    Applies ...

    Parameters
    ----------
    diag : :obj:`numpy.ndarray`
        Vector to be used for element-wise multiplication.
    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape

    Notes
    -----
    """

    def __init__(self, GD, GS, ET, k1, k2, coeff, dtype=None):
        self.GD = GD
        self.GS = GS
        self.ET = ET
        self.k1 = k1
        self.k2 = k2
        self.coeff = coeff
        self.N = GS.shape[1]
        self.Nrec = GS.shape[0]
        self.Ninc = self.ET.shape[1]
        super().__init__(dtype=np.dtype(dtype), shape=(2 * (self.N * self.Ninc + self.Nrec * self.Ninc), self.N))

    def _matvec(self, x):
        aux = np.multiply(np.matrix(self.ET), np.matrix(x).T)
        Ax = np.array(np.matmul(self.G, aux))
        return np.concatenate((np.imag(Ax), np.real(Ax)), axis=0).reshape(-1)

    def _rmatvec(self, y):
        y = y.reshape(2, -1).T
        y = np.matrix(y[:, 1] + 1j * y[:, 0])
        B = np.matmul(self.G.H, y.reshape(-1, self.Ninc))
        C = np.multiply(np.matrix(self.ET).H, B.T)
        C = np.sum(C, axis=0)
        C = np.matrix(C).T
        AHy = np.array(C).reshape(-1)

        return np.real(AHy)

    def compute_G(self):
        G = np.matrix(self.coeff * np.concatenate((self.k1 * self.GD, self.k2 * self.GS), axis=0))
        self.G = G

        return
