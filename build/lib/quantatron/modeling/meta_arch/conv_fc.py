import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from quantatron.config import configurable
from .build import META_ARCH_REGISTRY, move_device_like

@META_ARCH_REGISTRY.register()
class ConvFC(nn.Module):
    @configurable
    def __init__(self, length, column, hidden_dim, layer, output_dim):
        super(ConvFC, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, column), stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.fc = []
        dim = length*128
        for _ in range(layer-1):
            self.fc.append(nn.Linear(dim, hidden_dim))
            self.fc.append(nn.LeakyReLU())
            dim = hidden_dim
        self.fc.append(nn.Linear(hidden_dim, output_dim))

        self.fc = nn.Sequential(*(self.fc))

        self.loss = nn.MSELoss()

    @property
    def device(self):
        return self.conv1.weight.device

    def _move_to_current_device(self, sample):
        if isinstance(sample, Dict):
            for k in sample.keys():
                sample[k] = sample[k].to(self.device)
        if isinstance(sample, List):
            sample = [x.to(self.device) for x in sample]
        return sample

    def preprocess(self, sample):
        self._move_to_current_device(sample)
        if isinstance(sample, Dict):
            for k in sample.keys():
                sample[k] = sample[k].unsqueeze(1)
        if isinstance(sample, List):
            sample = [x.unsqueeze(1) for x in sample]
        return sample

    @classmethod
    def from_config(cls, cfg):
        return {
            "length": cfg.MODEL.LENGTH,
            "column": cfg.MODEL.COLUMN,
            "hidden_dim": cfg.MODEL.HIDDEN_DIM,
            "layer": cfg.MODEL.LAYER,
            "output_dim": cfg.MODEL.OUTPUT_DIM,
        }

    def forward(self, batched_inputs):
        batched_inputs = self.preprocess(batched_inputs)
        if not self.training:
            return self.inference(batched_inputs)
        data = F.leaky_relu(self.conv1(batched_inputs["daily"]))
        data = F.leaky_relu(self.conv2(data))
        data = data.flatten(1)
        pre = self.fc(data)

        loss = self.loss(pre, batched_inputs["gt"])
        return loss

    def inference(self, batched_inputs):
        data = F.leaky_relu(self.conv1(batched_inputs["daily"]))
        data = F.leaky_relu(self.conv2(data))
        data = data.flatten(1)
        pre = self.fc(data)
        return pre

