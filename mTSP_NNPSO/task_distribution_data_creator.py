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

# create task distribution data

# number of task situations

# generate task positions

# add task states

task_Data = torch.load("/home/supreet/NeuralPSO/mTSP_NNPSO/Dataset/taskConfig.pt")
print(task_Data[1])