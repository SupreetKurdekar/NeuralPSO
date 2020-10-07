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
from scipy.optimize import minimize

# data set up
taskData = torch.load("Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:2,:,:].float()
RobotStateBatch = robotData.repeat(2,1,1).float()

penalty = 2
# model
# model = vf.Allocator()
model = torch.load("D:\\NNPSO_Results\\DemoRun1\\InterimModel290.pth")


weightList,shapeList = utils.modelToWeightList(model)

# newWeightList = [i+1 for i in weightList]
# f1 = objectiveFunction.gradDescentObjFunc(weightList,penalty,RobotStateBatch.float(),TaskStateBatch.float())
# f2 = objectiveFunction.gradDescentObjFunc(newWeightList,penalty,RobotStateBatch.float(),TaskStateBatch.float())

# print(f1)
# print(f2)

res = minimize(objectiveFunction.gradDescentObjFunc,weightList,args=(penalty,RobotStateBatch.float(),TaskStateBatch.float()), method='SLSQP', options={'disp': True,'eps':0.00001})