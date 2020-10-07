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
from numpy import linalg as LA
import statistics


# taskData = torch.load("Dataset/taskConfig.pt")
# taskData = taskData.int()
# robotData = torch.load("Dataset/homeRobotState.pt")
# TaskStateBatch = taskData[0:2,:,:].float()
# RobotStateBatch = robotData.repeat(2,1,1).float()

# model = vf.Allocator()


def objectivefunction(model,RobotStateBatch,TaskStateBatch,k,max_iterations):
    stateFuncEvals = [0 for i in range(TaskStateBatch.shape[0])]
    stateConstViol = [0 for i in range(TaskStateBatch.shape[0])]
    # max_iterations = 10

    iterations = 0
    while torch.sum(TaskStateBatch[:,0]) > 0 and iterations < max_iterations:
        
        output = model(TaskStateBatch.float(),RobotStateBatch.float())
        Decision = output.squeeze().detach().numpy().astype(int)
        # print(Decision.shape)
        a = np.array([[1,1,1]]).T
        a = a[np.newaxis,:,:]
        Decision[:,-1,:][np.where(np.all(Decision == a,axis=1))] = 0

        b = np.array([[0,0,0]]).T
        b = b[np.newaxis,:,:]
        Decision[:,-1,:][np.where(np.all(Decision == b,axis=1))] = 1

        Decision = Decision.transpose(0,2,1)
        TaskStateBatch = TaskStateBatch.detach().numpy().astype(int).transpose(0,2,1)
        RobotStateBatch = RobotStateBatch.detach().numpy().astype(int).transpose(0,2,1)

        id = 0
        for case,task,robot in zip(Decision,TaskStateBatch,RobotStateBatch):
            if np.sum(task[:,0]) != 0:
                # print(case)
                # print(task)
                # print(robot)
                c = case.dot(1 << np.arange(case.shape[-1] - 1, -1, -1))-1
                # print(c)
                newRobotPositions = task[c,1:3]
                if np.all(newRobotPositions == robot[:,0:2]):
                    stateConstViol[id] = stateConstViol[id] + 3*k
                taskStatus = task[c,0]

                numberCompletedTasksVisited = len(taskStatus) - np.count_nonzero(taskStatus)
                stateConstViol[id] = stateConstViol[id] + numberCompletedTasksVisited

                numberOfRobotsToSameTask = len(c)-len(np.unique(c))
                stateConstViol[id] = stateConstViol[id] + numberOfRobotsToSameTask

                # update task status
                task[np.unique(c),0] = 0
                # update robot distance status
                robot[:,2] = robot[:,2] + LA.norm(task[c,1:3]-robot[:,0:2], axis=1)
                stateFuncEvals[id] = stateFuncEvals[id] + np.sum(LA.norm(task[c,1:3]-robot[:,0:2], axis=1))
                # update robot position status
                robot[:,0:2] = newRobotPositions
                # print(robot)
            id = id+1
        # print(TaskStateBatch)
        TaskStateBatch = torch.from_numpy(TaskStateBatch.transpose(0,2,1)).int()
        RobotStateBatch = torch.from_numpy(RobotStateBatch.transpose(0,2,1)).int()
        iterations = iterations + 1
    
    functionEvaluation = statistics.mean(stateFuncEvals)
    constraintViolation = statistics.mean(stateConstViol)
    return functionEvaluation,constraintViolation

def TaskStateDisplay(model,RobotStateBatch,TaskStateBatch,k,max_iterations):
    print("Initial Task State",TaskStateBatch[0][0])
    print("Initial Robot State",RobotStateBatch[0][0:2])
    stateFuncEvals = [0 for i in range(TaskStateBatch.shape[0])]
    stateConstViol = [0 for i in range(TaskStateBatch.shape[0])]
    # max_iterations = 10

    iterations = 0
    while torch.sum(TaskStateBatch[:,0]) > 0 and iterations < max_iterations:
        
        output = model(TaskStateBatch.float(),RobotStateBatch.float())
        Decision = output.squeeze().detach().numpy().astype(int)
        # print(Decision.shape)
        a = np.array([[1,1,1]]).T
        a = a[np.newaxis,:,:]
        Decision[:,-1,:][np.where(np.all(Decision == a,axis=1))] = 0

        b = np.array([[0,0,0]]).T
        b = b[np.newaxis,:,:]
        Decision[:,-1,:][np.where(np.all(Decision == b,axis=1))] = 1

        Decision = Decision.transpose(0,2,1)
        TaskStateBatch = TaskStateBatch.detach().numpy().astype(int).transpose(0,2,1)
        RobotStateBatch = RobotStateBatch.detach().numpy().astype(int).transpose(0,2,1)

        id = 0
        for case,task,robot in zip(Decision,TaskStateBatch,RobotStateBatch):
            if np.sum(task[:,0]) != 0:
                # print(case)
                # print(task)
                # print(robot)
                c = case.dot(1 << np.arange(case.shape[-1] - 1, -1, -1))-1
                # print(c)
                newRobotPositions = task[c,1:3]
                if np.all(newRobotPositions == robot[:,0:2]):
                    stateConstViol[id] = stateConstViol[id] + 3*k
                taskStatus = task[c,0]

                numberCompletedTasksVisited = len(taskStatus) - np.count_nonzero(taskStatus)
                stateConstViol[id] = stateConstViol[id] + numberCompletedTasksVisited

                numberOfRobotsToSameTask = len(c)-len(np.unique(c))
                stateConstViol[id] = stateConstViol[id] + numberOfRobotsToSameTask

                # update task status
                task[np.unique(c),0] = 0
                # update robot distance status
                robot[:,2] = robot[:,2] + LA.norm(task[c,1:3]-robot[:,0:2], axis=1)
                stateFuncEvals[id] = stateFuncEvals[id] + np.sum(LA.norm(task[c,1:3]-robot[:,0:2], axis=1))
                # update robot position status
                robot[:,0:2] = newRobotPositions
                # print(robot)
            id = id+1
        # print(TaskStateBatch)
        TaskStateBatch = torch.from_numpy(TaskStateBatch.transpose(0,2,1)).int()
        RobotStateBatch = torch.from_numpy(RobotStateBatch.transpose(0,2,1)).int()
        print("Task State",TaskStateBatch[0][0])
        print("Robot State",RobotStateBatch[0][0:2])
        iterations = iterations + 1
    
    functionEvaluation = statistics.mean(stateFuncEvals)
    constraintViolation = statistics.mean(stateConstViol)
    return functionEvaluation,constraintViolation

def gradDescentObjFunc(weightList,penalty,RobotStateBatch,TaskStateBatch):
    model = vf.Allocator()
    # weightList = [float(item) for item in weightList]
    # model = utils.updateModelWithNewWeightList(weightList,model)
   
    # func,conv = objectivefunction(model,RobotStateBatch.float(),TaskStateBatch.float(),5,10)
    # print(func)
    # print(conv)

    weightList = [float(item) for item in weightList]
    model = utils.updateModelWithNewWeightList(weightList,model)
    func,conv = objectivefunction(model,RobotStateBatch.float(),TaskStateBatch.float(),5,10)
    print(func)
    return func + penalty*conv

