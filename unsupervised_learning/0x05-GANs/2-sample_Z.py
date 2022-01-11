#!/usr/bin/env python3
"""
contains function sample_Z()
"""

import torch

def sample_Z(mu, sigma, sampleType, dInputSize, gInputSize, mbatchSize=None):
    """
    creates input for the generator and discriminator
    """
    if sampleType == 'G':
        gCreate = torch.randn((dInputSize, gInputSize))
        return gCreate
    elif sampleType == 'D':
        dCreate = torch.normal(mu, sigma, (mbatchSize, dInputSize))
        return dCreate
    else:
        return 0
