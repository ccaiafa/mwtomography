#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        current_image = self.images[idx]
        relperm_minus_one = current_image.get_relative_permittivities() - 1.0
        if self.transform:
            relperm_minus_one = self.transform(relperm_minus_one)
        return idx, relperm_minus_one

