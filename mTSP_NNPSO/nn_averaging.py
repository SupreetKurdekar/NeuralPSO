# import utils
import pyDOE
import numpy as np
from pyDOE import lhs
import random
# import fitness_function
import pandas as pd
import multiprocessing as mp
from matplotlib import animation
from matplotlib import pyplot as plt
import time as tm
import copy
import singleObjFuncs

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import visual_functions as vf
import utils
import sys

# data set up
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)


X_ = np.expand_dims(X,axis=2)
Y_ = np.expand_dims(Y,axis=2)
data = np.concatenate((X_,Y_),axis=2)
data = data.reshape(900,2)
# dataTensor = torch.from_numpy(data)
dataTensor = torch.Tensor(data)
# print(dataTensor.type)

truth = vf.paraboloid(X, Y)/70
truth_ = truth.reshape(900, 1)
truth_ = torch.tensor(truth_)

# model parts
criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

numNetworks = 50
num_iterations = 50
listOfNetworks = []
for i in range(numNetworks):
    listOfNetworks.append(vf.Perceptron())

alpha = 0.1
BetaLocal = 0.2
BetaGlobal = 0.2

PersonalBestFuncEvals = [sys.float_info.max for i in range(len(listOfNetworks))]
PersonalBestCVs = [float(1000000000000000) for i in range(len(listOfNetworks))]
personalbestNetworks = copy.deepcopy(listOfNetworks)
velocities = copy.deepcopy(listOfNetworks)

for velocity in velocities:
    for trial in velocity.parameters():
        trial.data = trial.data*0
        # print(trial.data)
# global best initialisation
GbCv = 1000000000000001
GbF = sys.float_info.max*2
GbN = vf.Perceptron()

iteration = 0
while iteration < num_iterations:
    # local best update
    id = 0
    for network,PbF,PbCv,PbN in zip(listOfNetworks,PersonalBestFuncEvals,PersonalBestCVs,personalbestNetworks):
        output = network(dataTensor)
        funcEval = criterion(output.float(), truth_.float())
        CViolation = float(0)
        count = 0
        # funcEval,CViolation = criterion()objectiveFcn(network)
        if CViolation < PbCv: 
            PersonalBestCVs[id] = CViolation
            PersonalBestFuncEvals[id] = funcEval.item()
            personalbestNetworks[id] = listOfNetworks[id]
            
        elif CViolation == PbCv and funcEval.item() < PbF:
            count += 1
            PersonalBestCVs[id] = CViolation
            PersonalBestFuncEvals[id] = funcEval.item()
            personalbestNetworks[id] = listOfNetworks[id]
        id = id + 1
    # codesmell -
    # global best update
    id = 0
    for network,PbF,PbCv,PbN in zip(listOfNetworks,PersonalBestFuncEvals,PersonalBestCVs,personalbestNetworks):
        if PbCv < GbCv: 
            GbCv = PersonalBestCVs[id]
            GbF = PersonalBestFuncEvals[id]
            GbN = personalbestNetworks[id]
            
        elif CViolation == PbCv and PbF < GbF:
            GbCv = PersonalBestCVs[id]
            GbF = PersonalBestFuncEvals[id]
            GbN = personalbestNetworks[id]
        
        id = id + 1

    for network,velocity,PbN in zip(listOfNetworks,velocities,personalbestNetworks):
        for GbNparams,networkParams,velocityParams, PbParam in zip(GbN.parameters(),network.parameters(),velocity.parameters(), PbN.parameters()):
            velocityParams.data *= alpha
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocityParams.data += r1*BetaLocal*(PbParam.data - networkParams.data) + r2*BetaGlobal*(GbNparams.data-networkParams.data)
            # params1.data += params2.data * beta
            # print(params1)

    for network,velocity in zip(listOfNetworks,velocities):
        for networkParams,velocityParams in zip(network.parameters(),velocity.parameters()):
            networkParams.data = networkParams.data + velocityParams.data

    print(GbF,count)
    iteration += 1
