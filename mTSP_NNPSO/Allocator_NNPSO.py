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
import statistics
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import visual_functions as vf
import utils
import sys
import objectiveFunction
# data set up
taskData = torch.load("Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:2,:,:].float()
RobotStateBatch = robotData.repeat(2,1,1).float()


numNetworks = 50
num_iterations = 50
listOfNetworks = []
for i in range(numNetworks):
    listOfNetworks.append(vf.Allocator())

alpha = 1
BetaLocal = 2
BetaGlobal = 2

PersonalBestFuncEvals = [sys.float_info.max for i in range(len(listOfNetworks))]
PersonalBestCVs = [int(1000000000000000) for i in range(len(listOfNetworks))]
personalbestNetworks = copy.deepcopy(listOfNetworks)
velocities = copy.deepcopy(listOfNetworks)

for velocity in velocities:
    for trial in velocity.parameters():
        trial.data = trial.data*0
        # print(trial.data)
# global best initialisation
GbCv = int(1000000000000000)
GbF = sys.float_info.max*2
GbN = vf.Allocator()
GlobalBesthistory = []
GlobalMeanHistory = []
iteration = 0
while iteration < num_iterations:
    TaskStateBatch = taskData[0:2,:,:].float()
    RobotStateBatch = robotData.repeat(2,1,1).float()
    # local best update
    id = 0
    for network,PbF,PbCv,PbN in zip(listOfNetworks,PersonalBestFuncEvals,PersonalBestCVs,personalbestNetworks):
 
        funcEval,CViolation = objectiveFunction.objectivefunction(network,RobotStateBatch,TaskStateBatch,5,10)
        # funcEval,CViolation = criterion()objectiveFcn(network)
        if CViolation < PbCv: 
            PersonalBestCVs[id] = CViolation
            PersonalBestFuncEvals[id] = funcEval
            personalbestNetworks[id] = listOfNetworks[id]
            
        elif CViolation == PbCv and funcEval < PbF:
            PersonalBestCVs[id] = CViolation
            PersonalBestFuncEvals[id] = funcEval
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

    print("Global",GbF,GbCv)
    print("Avg",statistics.mean(PersonalBestFuncEvals),statistics.mean(PersonalBestCVs))
    iteration += 1
