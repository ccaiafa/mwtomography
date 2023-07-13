#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pylops import LinearOperator
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class CSoperator(LinearOperator):
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

    def __init__(self, GD, GS, ET, k1, k2, D, coeff, dtype=None):
        self.GD = GD
        self.GS = GS
        self.ET = ET
        self.k1 = k1
        self.k2 = k2
        self.D = D
        self.coeff = coeff
        self.N = GS.shape[1]
        self.Nrec = GS.shape[0]
        self.Ninc = self.ET.shape[1]
        self.Nat = self.D.shape[1]
        super().__init__(dtype=np.dtype(dtype), shape=(2 * (self.N * self.Ninc + self.Nrec * self.Ninc), self.Nat))

    def _matvec(self, x):
        x = x / self.norm2col
        x = np.matmul(self.D, x)
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
        AHy = np.array(np.multiply(np.matrix(1/self.norm2col).T, np.matmul(self.D.T, C))).reshape(-1)

        return np.real(AHy)

    def norm2col_op(self):
        G = np.matrix(self.coeff * np.concatenate((self.k1 * self.GD, self.k2 * self.GS), axis=0))

        # compute norm of columns
        R = np.multiply(np.matmul(G.H, G), np.matmul(self.ET, np.matrix(self.ET).H))
        DTR = np.matmul(np.transpose(self.D), R)
        DTR = np.multiply(self.D, np.transpose(DTR))
        norm2col = np.sqrt(np.sum(DTR.real, axis=0).T)
        self.norm2col = np.array(norm2col).reshape(-1)
        self.G = G

        return
