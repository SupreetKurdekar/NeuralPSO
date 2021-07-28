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
# import the os module
import os
import shutil

# data set up

# number of task distributions to use
batch_size = 900

taskData = torch.load("mTSP_NNPSO/Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("mTSP_NNPSO/Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:batch_size,:,:].float()
RobotStateBatch = robotData.repeat(batch_size,1,1).float()

# detect the current working directory and print it
path = os.getcwd()
print ("The current working directory is %s" % path)

run_name = "900_tasks"
path = "/home/supreet/NeuralPSO/mTSP_NNPSO/results"
newPath = os.path.join(path,run_name)
# os.mkdir(newPath)
dir = newPath
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

numNetworks = 80
num_iterations = 500
listOfNetworks = []
for i in range(numNetworks):
    listOfNetworks.append(vf.Allocator())

alpha = 0.8
BetaLocal = 1.2
BetaGlobal = 1
StagnationPenalty = 5
InternalIterations = 10

newPath2 = os.path.join(newPath,"parameters.txt")
OptimParameters = {"Num_Networks":numNetworks,"num_iterations":num_iterations,"alpha":alpha,"BetaLocal":BetaLocal,"BetaGlobal":BetaGlobal,"StagnationPenalty":StagnationPenalty,"InternalIterations":InternalIterations}

with open(newPath2, 'w') as f:
    print(OptimParameters, file=f)

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
    TaskStateBatch = taskData[0:batch_size,:,:].float()
    RobotStateBatch = robotData.repeat(batch_size,1,1).float()
    # local best update
    id = 0
    for network,PbF,PbCv,PbN in zip(listOfNetworks,PersonalBestFuncEvals,PersonalBestCVs,personalbestNetworks):
        TaskStateBatch = taskData[0:batch_size,:,:].float()
        RobotStateBatch = robotData.repeat(batch_size,1,1).float()
        funcEval,CViolation = objectiveFunction.objectivefunction(network,RobotStateBatch,TaskStateBatch,StagnationPenalty,InternalIterations)
        # funcEval,CViolation = criterion()objectiveFcn(network)
        if CViolation < PbCv: 
            PersonalBestCVs[id] = copy.deepcopy(CViolation)
            PersonalBestFuncEvals[id] = copy.deepcopy(funcEval)
            personalbestNetworks[id] = copy.deepcopy(listOfNetworks[id])
            
        elif CViolation == PbCv and funcEval < PbF:
            PersonalBestCVs[id] = copy.deepcopy(CViolation)
            PersonalBestFuncEvals[id] = copy.deepcopy(funcEval)
            personalbestNetworks[id] = copy.deepcopy(listOfNetworks[id])
        id = id + 1
    # codesmell -
    # global best update
    id = 0
    for network,PbF,PbCv,PbN in zip(listOfNetworks,PersonalBestFuncEvals,PersonalBestCVs,personalbestNetworks):
        if PbCv < GbCv: 
            GbCv = copy.deepcopy(PersonalBestCVs[id])
            GbF = copy.deepcopy(PersonalBestFuncEvals[id])
            GbN = copy.deepcopy(personalbestNetworks[id])
            
        elif GbCv == PbCv and PbF < GbF:
            GbCv = copy.deepcopy(PersonalBestCVs[id])
            GbF = copy.deepcopy(PersonalBestFuncEvals[id])
            GbN = copy.deepcopy(personalbestNetworks[id])
        
        id = id + 1
    
    # for personalBestFunc,personalbestCV in zip(PersonalBestFuncEvals,PersonalBestCVs):

    # print("Check1")
    print(objectiveFunction.objectivefunction(GbN,RobotStateBatch,TaskStateBatch,5,10))
    # print(GbCv)
    for network,velocity,PbN in zip(listOfNetworks,velocities,personalbestNetworks):
        for GbNparams,networkParams,velocityParams, PbParam in zip(GbN.parameters(),network.parameters(),velocity.parameters(), PbN.parameters()):
            velocityParams.data *= alpha
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocityParams.data += r1*BetaLocal*(PbParam.data - networkParams.data) + r2*BetaGlobal*(GbNparams.data-networkParams.data)
            # params1.data += params2.data * beta
            # print(params1)
    # print("check")
    # print(objectiveFunction.objectivefunction(GbN,RobotStateBatch,TaskStateBatch,5,10))
    # print(GbCv)
    for network,velocity in zip(listOfNetworks,velocities):
        for networkParams,velocityParams in zip(network.parameters(),velocity.parameters()):
            networkParams.data = networkParams.data + velocityParams.data

            # clipping the weights to remain between range
            networkParams.data = torch.clamp(networkParams.data,-1000,1000)

    # print("Global",GbF,GbCv)
    # print(objectiveFunction.objectivefunction(GbN,RobotStateBatch,TaskStateBatch,5,10))
    GlobalBesthistory.append((GbF,GbCv))
    # print("Avg",statistics.mean(PersonalBestFuncEvals),statistics.mean(PersonalBestCVs))
    GlobalMeanHistory.append((statistics.mean(PersonalBestFuncEvals),statistics.mean(PersonalBestCVs)))

    if iteration % 10 == 0:
        torch.save(GbN,os.path.join(newPath,"InterimModel"+str(iteration)+".pth"))
        with open(os.path.join(newPath,str(iteration)+"GlobalConvergence.txt"), 'w') as fp:
            fp.write('\n'.join('%s %s' % x for x in GlobalBesthistory))
        with open(os.path.join(newPath,str(iteration)+"MeanConvergence.txt"), 'w') as fp:
            fp.write('\n'.join('%s %s' % x for x in GlobalMeanHistory))

    iteration += 1

torch.save(GbN,os.path.join(newPath,"FinalGlobalBestModel.pth"))

with open(os.path.join(newPath,"GlobalConvergence.txt"), 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in GlobalBesthistory))
with open(os.path.join(newPath,"MeanConvergence.txt"), 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in GlobalMeanHistory))