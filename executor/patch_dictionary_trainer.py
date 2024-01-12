#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch

from configs.constants import Constants
from configs.logger import Logger
from mwtomography.dataloader.image_dataset import ImageDataset
from torch.utils.data import random_split, DataLoader

from sklearn.decomposition import DictionaryLearning

from mwtomography.utils import FileManager
from mwtomography.utils import Plotter

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/trainer/dictionary/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Dictionary_Trainer:
    def __init__(self, images_path_prefix, no_of_pixels, checkpoint_path_prefix):
        basic_parameters = Constants.get_basic_parameters()
        dict_learn_parameters = basic_parameters["patch_dict_learn"]
        LOG.info("Starting trainer in standard mode")
        self.checkpoint_path = ROOT_PATH + checkpoint_path_prefix + "trained_dict.pt"
        self.params = dict_learn_parameters
        images_path = ROOT_PATH + images_path_prefix + "images" + "_" + str(no_of_pixels) + "x" + str(
            no_of_pixels) + ".pkl"
        self.load_datasets(images_path)
        self.plotter = Plotter()
        self.no_of_pixels = no_of_pixels

    def dict_train(self, load, plot_interval, training_logs_plots_path_prefix, validation_logs_plots_path_prefix,
                   error_logs_plots_path_prefix):
        init_epoch = 0
        min_valid_loss = np.inf
        training_errors = OrderedDict()
        validation_errors = OrderedDict()
        start_epoch_time = datetime.now()
        time_elapsed = start_epoch_time - start_epoch_time
        if load:
            LOG.info(f'''Going to load model from {self.checkpoint_path}''')
            # self.unet, self.optimizer, init_epoch, min_valid_loss, training_errors, validation_errors, time_elapsed = \
            #    CheckpointManager.load_checkpoint(self.unet, self.checkpoint_path, self.device,
            #                                      optimizer=self.optimizer)

        LOG.info(f'''Starting training dictionary:
                            N components:           {self.params["n_components"]}
                            Alpha:                  {self.params["alpha"]}
                            Max iter:               {self.params["max_iter"]}
                            Tol:                    {self.params["tol"]}
                            Fit algorithm:          {self.params["fit_algorithm"]}
                            Transform algorithm:    {self.params["transform_algorithm"]}
                            Transform alpha:        {self.params["transform_alpha"]}  
                            Verbose:                {self.params["verbose"]}
                            Num epochs:             {self.params["num_epochs"]}                                                            
                            Random state:           {self.params["random_state"]}
                            Transf Max iter:        {self.params["transform_max_iter"]}
                            Batch size:             {self.params["batch_size"]}
                            Shuffle:                {self.params["shuffle"]}
                            Training size:   {len(self.train_loader.dataset)}
                            Validation size: {len(self.val_loader.dataset)}
                            Time elapsed:    {time_elapsed}      
                        ''')

        dict_learner = DictionaryLearning(
            n_components=self.params["n_components"],
            alpha=self.params["alpha"],
            max_iter=self.params["max_iter"],
            fit_algorithm=self.params["fit_algorithm"],
            transform_algorithm=self.params["transform_algorithm"],
            transform_alpha=self.params["transform_alpha"],
            transform_max_iter=self.params["transform_max_iter"],
            random_state=self.params["random_state"],
            verbose=self.params["verbose"])

        #filename_dict = os.path.join(ROOT_PATH + "/dictionary/trained_dict_epoch_1")
        #aux = FileManager.load(filename_dict)
        #dict_learner.set_params(dict_init=aux.components_)

        for epoch in range(init_epoch + 1, self.params["num_epochs"] + 1):
            training_loss = 0.0
            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.params["num_epochs"]}',
                      unit='img') as pbar:
                for ix, rel_perm_minus_one in self.train_loader:
                #for ix, rel_perm_minus_one in enumerate(self.train_loader):


                    start_batch_time = datetime.now()
                    X = self.extract_nonoverlapped_patches(rel_perm_minus_one)
                    #X = rel_perm_minus_one.reshape((rel_perm_minus_one.shape[0], -1))
                    code = dict_learner.fit_transform(X)
                    sq_error = np.mean(np.sum((X.numpy() - code @ dict_learner.components_) ** 2, axis=1) / np.sum(X.numpy() ** 2, axis=1))  # np.linalg.norm(X - Xap, 'fro')
                    dict_learner.set_params(dict_init=dict_learner.components_, code_init=code)

                    training_loss += sq_error

                    pbar.update()
                    pbar.set_postfix(**{'squared error (batch)': sq_error})

                    filename_dict = os.path.join(ROOT_PATH + "/dictionary/trained_dict_epoch_" + "_patch_" + str(self.no_of_pixels) + "x" + str(self.no_of_pixels) + "_epoch_" + str(epoch) + "_batch_size_" + str(self.params["batch_size"]) + "_ncomps_" + str(self.params["n_components"]) +".pkl")
                    FileManager.save(dict_learner, filename_dict)

                    time_elapsed += (datetime.now() - start_batch_time)
                    print("Batch elapsed time=" + str(time_elapsed) + "s")


            training_loss = training_loss / len(self.train_loader.dataset)
            training_errors[epoch] = training_loss
            #validation_loss = self.validate(epoch, validation_logs_plots_path_prefix)
            #validation_errors[epoch] = validation_loss

            #LOG.info(f'''Statistics of epoch {epoch}/{self.num_epochs}:
            #                    Training loss: {training_loss:.2E}
            #                    Validation loss: {validation_loss:.2E}
            #                    Min validation loss: {min_valid_loss:.2E}''')
            time_elapsed += (datetime.now() - start_epoch_time)
            start_epoch_time = datetime.now()
            #if min_valid_loss > validation_loss:
            #    min_valid_loss = validation_loss
            #    LOG.info(
            #        f'''Saving progress for epoch {epoch} with loss {validation_loss:.2E} to path {self.checkpoint_path}''')
            #    CheckpointManager.save_checkpoint(self.unet, self.optimizer, self.checkpoint_path, epoch,
            #                                      min_valid_loss, training_errors, validation_errors, time_elapsed)
            #else:
            #    LOG.info(f'''Updating checkpoint with new epoch value ({epoch}) in path {self.checkpoint_path}''')
            #    CheckpointManager.update_epoch(self.checkpoint_path, epoch, training_errors, validation_errors,
            #                                   time_elapsed)
        LOG.info(f'''Finishing training of the network''')
        LOG.info(f'''Total duration of the training was {time_elapsed}''')

        if test:
            path = ROOT_PATH + error_logs_plots_path_prefix + "test_errors_{:%Y-%m-%d_%H:%M:%S}.png".format(
                datetime.now())
        else:
            path = ROOT_PATH + error_logs_plots_path_prefix + "errors_{:%Y-%m-%d_%H:%M:%S}.png".format(datetime.now())

        LOG.info(f'''Saving per epoch training/validation errors plot to path {path}''')
        self.plotter.plot_errors("Model Loss", training_errors, "Training error", validation_errors,
                                 "Validation error", "epoch", path)

    def load_datasets(self, images_path):
        LOG.info("Loading images from file %s", images_path)
        images = ImageDataset(FileManager.load(images_path))
        LOG.info("%d images loaded", len(images))
        self.n_val = int(len(images) * self.params["validation_proportion"])
        self.n_train = len(images) - self.n_val
        train_set, val_set, _ = random_split(images, [self.n_train, self.n_val, 0],
                                             generator=torch.Generator().manual_seed(self.params["manual_seed"]))
        LOG.info("Train set has %d images", self.n_train)
        LOG.info("Validation set has %d images", self.n_val)

        # loader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = DataLoader(train_set, shuffle=False, batch_size=self.params["batch_size"])
        self.val_loader = DataLoader(val_set, shuffle=True, batch_size=self.params["batch_size"])

    def validate(self, epoch, validation_logs_plots_path_prefix):
        LOG.info(f'''Validating model for epoch {epoch}''')
        validation_loss = 0.0
        self.unet.eval()
        for ix, (images, labels) in enumerate(self.val_loader):
            images = images.to(device=self.device, dtype=torch.float32)
            labels = labels.to(device=self.device, dtype=torch.float32)
            prediction = self.unet(images)
            loss = self.criterion(prediction, labels)
            validation_loss += loss.item()

            if ix % 5 == 0:
                plot_title = "Validation - Epoch {} - Batch {}".format(epoch, ix)
                path = ROOT_PATH + validation_logs_plots_path_prefix + "validation_image_{}_{}.png".format(epoch,
                                                                                                           ix)
                LOG.info(f'''Saving validation image plot to path {path}''')
                self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                          images, prediction, loss.item())
        return validation_loss / len(self.val_loader.dataset)

    def extract_nonoverlapped_patches(self, x):
        # Input is [N, 64, 64]
        N = x.shape[0]
        P = x.shape[1] # 64
        J = 16 # patch size = 16
        n_blocks = P//J # n_blocs = 4
        x = x.reshape(N, P, n_blocks, J) # [N, 64, 4, 16]
        x = x.permute(0, 2, 3, 1) # [N, 4, 16, 64]
        x = x.reshape(N, n_blocks, J, n_blocks, J) # [N, 4, 16, 4, 16]
        x = x.permute(4, 2, 0, 3, 1) # [16, 16, N, 4, 4]
        x = x.reshape(J, J, n_blocks*n_blocks*N) # [16, 16, 4*4*N]
        x = x.permute(2, 0, 1) # [4*4*N, 16,16]
        x = x.reshape(n_blocks*n_blocks*N, J*J) # [4*4*N, 16*16]

        # remove zero slices
        idx = torch.nonzero(x.sum(dim=1))
        x = x[idx]
        return x.squeeze()
