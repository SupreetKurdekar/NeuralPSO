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


k = 5
model = vf.Allocator()

# data set up
taskData = torch.load("Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:2,:,:].float()
RobotStateBatch = robotData.repeat(2,1,1).float()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
funcEval,ConstViolation = objectiveFunction.objectivefunction(model,RobotStateBatch,TaskStateBatch,5,10)
print(funcEval,ConstViolation)
fval = torch.tensor(float(funcEval + k*ConstViolation),requires_grad=True)
zero = torch.tensor(0.0,requires_grad=True)
loss = criterion(fval,zero)
# print(zero)
# Backward pass
loss.backward()
optimizer.step()
print(funcEval,ConstViolation)

# print(fval)