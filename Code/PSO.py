import torch
import torch.nn as nn
import torch.nn.utils as nutils
import numpy as np
import torch

# Problem setup parameters

def pso(network,config):

    nets = [network.Net() for i in range(config.pso["Number of Particles"])]

    bestParticles = [nutils.parameters_to_vector(net.parameters()) for net in nets ]
    bestParticles = torch.stack(bestParticles)


    personal_best_evals = [1000000000000000.0]*len(nets)
    global_best_eval = 10000000000000000.0
    global_best_id = 0
    funcEvals = []

    # do gradient based training
    for net in nets:
        funcEvals.append(network.train(net,network.train_config))

    currentParticles = []
    # update personal best function evals and networks
    # update the global best network id and the global best function value
    for id,oldEval,newEval,particle,net in enumerate(zip(personal_best_evals,funcEvals,bestParticles,nets)):
        currentParticles.append(nutils.parameters_to_vector(net.parameters()))
        if newEval < oldEval:
            oldEval = newEval
            particle = nutils.parameters_to_vector(net.parameters())

        if newEval < global_best_eval:
            global_best_eval = newEval
            global_best_id = id

    currentParticles = torch.stack(currentParticles)

    velocity = config.pso["alpha"]*currentParticles + config.pso["Local Beta"]*torch.rand(1)*bestParticles + config.pso["Gamma"]*torch
    

    

    

    


    
    return 0