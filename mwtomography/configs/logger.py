#!/usr/bin/env python
""" wrapper around logging module """
import os
import logging


class Logger:

    @staticmethod
    def get_root_logger(logger_name, filename=None):
        """ get the logger object """
        logger = logging.getLogger(logger_name)
        debug = os.environ.get('ENV', 'development') == 'development'
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        ch_ = logging.StreamHandler()
        ch_.setFormatter(formatter)
        logger.addHandler(ch_)

        if filename:
            fh_ = logging.FileHandler(filename)
            fh_.setFormatter(formatter)
            logger.addHandler(fh_)

        return logger

    @staticmethod
    def get_child_logger(root_logger, name):
        return logging.getLogger('.'.join([root_logger, name]))
