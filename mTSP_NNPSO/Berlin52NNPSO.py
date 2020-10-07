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

Data = np.loadtxt("ACO\Berlin52.txt").astype(int)
binStr = utils.vec_bin_array(Data[:,0],6)
Data = np.hstack((Data,binStr))
ones = np.ones(len(Data))
Data[:,0] = ones

Task = torch.Tensor(Data).unsqueeze(dim=0)
Task = Task.permute(0,2,1)
print(Task.size())
# print(Task)
robotData = torch.load("Dataset/homeRobotState.pt")
RobotStateBatch = robotData.repeat(1,1,1).float()
print(RobotStateBatch.shape)
print(RobotStateBatch)


model = vf.Allocator2()
output = model(Task,RobotStateBatch)

print(output)