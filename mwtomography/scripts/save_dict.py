#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import hdf5storage
import sys
from os.path import join as pjoin

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_manager import FileManager


if __name__ == "__main__":
    filename_dict_learner = os.path.join(
        ROOT_PATH + "/dictionary/trained_dict_epoch__patch_64x64_epoch_4_batch_size_125000_ncomps_1024.pkl")
    aux = FileManager.load(filename_dict_learner)

    filename_dict = os.path.join(
        ROOT_PATH + "/dictionary/trained_dict_epoch__patch_64x64_epoch_4_batch_size_125000_ncomps_1024_dict.pkl")
    FileManager.save(aux.components_.transpose(), filename_dict)


