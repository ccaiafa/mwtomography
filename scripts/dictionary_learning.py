#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.constants import Constants

#from executor.dictionary_trainer import Dictionary_Trainer
from executor import Dictionary_Trainer

#from executor.kronecker_dictionary_trainer import Dictionary_Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Load latest checkpoint", action='store_true')
    args = parser.parse_args()
    load = args.load

    images_parameters = Constants.get_basic_parameters()["images"]
    images_path_prefix = "/data/image_generator/"
    no_of_pixels = images_parameters["no_of_pixels"]
    checkpoint_path_prefix = "/data/trainer/dictionary/"
    dict_training_logs_plots_path_prefix = "/logs/trainer/dictionary/training_images/"
    dict_validation_logs_plots_path_prefix = "/logs/trainer/dictionary/validation_images/"
    error_logs_plots_path_prefix = "/logs/dict_trainer/dictionary/"
    plot_interval = 50
    trainer = Dictionary_Trainer(images_path_prefix, no_of_pixels, checkpoint_path_prefix)
    trainer.dict_train(load, plot_interval, dict_training_logs_plots_path_prefix,
                       dict_validation_logs_plots_path_prefix,
                       error_logs_plots_path_prefix)
