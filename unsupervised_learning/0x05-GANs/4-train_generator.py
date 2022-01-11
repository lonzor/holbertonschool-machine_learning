#!/usr/bin/env python3
"""
contains function train_gen()
"""
import torch

sample_Z = __import__('2-sample_Z').sample_Z


def train_gen(Gen, Dis, gInputSize, dInputSize, mbatchSize, steps, optimizer, crit):
    """
    function should return the two item methods that belong to loss entropy class. 
    for real and fake
    """
    for i in range(steps):
        real_samps_labs = torch.ones((mbatchSize, 1))

        gen_lst = []

        for k in range(mbatchSize):
            lat_samps = sample_Z(0.0, 1.0, 'G', dInputSize, gInputSize)
            gen_samps = Gen(lat_samps)
            gen_lst.append(gen_samps)

        gen_samps = torch.stack(gen_lst, axis=0).reshape(mbatchSize, dInputSize)
        dOutput = Dis(gen_samps)
        gLoss = crit(dOutput, real_samps_labs)
        gLoss.backward()
        optimizer.step()

    return gLoss, gen_samps
