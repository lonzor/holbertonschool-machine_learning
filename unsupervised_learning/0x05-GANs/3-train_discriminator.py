#!/usr/bin/env python3
"""
contains function train_dis()
"""
import torch

sample_Z = __import__('2-sample_Z').sample_Z


def train_dis(Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    function should return the two item methods that belong to loss entropy class. 
    for real and fake
    """

    sample_Z = __import__('2-sample_Z').sample_Z

    for i in range(steps):
        real_samps = sample_Z(0.0, 1.0, 'D', dInputSize, gInputSize, mbatchSize=mbatchSize)
        real_samps_labs = torch.ones((mbatchSize, 1))

        gen_lst = []

        for k in range(mbatchSize):
            lat_samps = sample_Z(0.0, 1.0, 'G', dInputSize, gInputSize)
            gen_samps = Gen(lat_samps)
            gen_lst.append(gen_samps)

        gen_samps = torch.stack(gen_lst, axis=0).reshape(mbatchSize, dInputSize)
        gen_samps_labs = torch.zeros((mbatchSize, 1))
        all_samps = torch.cat((real_samps, gen_samps))
        all_samps_labs = torch.cat((real_samps_labs, gen_samps_labs))

        Dis.zero_grad()
        dOutput = Dis(all_samps)
        dLoss = crit(dOutput, all_samps_labs)
        dLoss.backward()
        optimizer.step()

    return dLoss, all_samps
