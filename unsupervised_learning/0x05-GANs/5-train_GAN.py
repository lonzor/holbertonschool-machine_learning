#!/usr/bin/env python3
"""
contains function train_gan()
"""

import torch

train_generator = __import__('4-train_generator').train_gen
train_discriminator = __import__('3-train_discriminator').train_dis
Generator = __import__('0-generator').Generator(1,16,1)
Discriminator = __import__('1-discriminator').Discriminator(dInputSize,16,1)
loss = torch.nn.BCELoss()

