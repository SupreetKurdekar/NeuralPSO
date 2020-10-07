import torch
import visual_functions as vf
import numpy as np
from numpy import linalg as LA

taskData = torch.load("Dataset/taskConfig.pt")
taskData = taskData.int()
# print(taskData.dtype)
# print(taskData.shape)
testTaskData = taskData[0:200]
trainTaskData = taskData[200:]

robotData = torch.load("Dataset/homeRobotState.pt")
# print("robot",robotData)
taskbatch = trainTaskData[0:2].float()
robotbatch = robotData.repeat(2,1,1).float()
# print(robotbatch.shape)
# print(robotbatch[0])
k = 3
model = vf.Allocator()
ModelCv = 0
listCv = []
for i in range(10):
    output = model(taskbatch,robotbatch.float())
    Decision = output.squeeze().detach().numpy().astype(int)
    # print(Decision)
    a = np.array([[1,1,1]]).T
    a = a[np.newaxis,:,:]
    Decision[:,-1,:][np.where(np.all(Decision == a,axis=1))] = 0

    b = np.array([[0,0,0]]).T
    b = b[np.newaxis,:,:]
    Decision[:,-1,:][np.where(np.all(Decision == b,axis=1))] = 1

    # print(Decision)
    for case,task,robot in zip(Decision,taskbatch,robotbatch):
        # case = case.transpose()
        task = task.detach().numpy().astype(int).transpose()
        robot = robot.detach().numpy().astype(int).transpose()
        if np.sum(task[:,0]) != 0:
            print(case)
            print(task)
            print(robot)
            c = case.dot(1 << np.arange(case.shape[-1] - 1, -1, -1))-1
            newRobotPositions = task[c,1:3]
            if newRobotPositions == robot[:,0:2]:
                cv = cv + 3*k
            taskStatus = task[c,0]

            numberCompletedTasksVisited = len(taskStatus) - np.count_nonzero(taskStatus)
            cv = cv + numberCompletedTasksVisited

            numberOfRobotsToSameTask = len(c)-len(np.unique(c))
            cv = cv + numberOfRobotsToSameTask

            # update task status
            task[np.unique(c),0] = 0
            # update robot distance status
            robot[:,2] = robot[:,2] + LA.norm(task[c,1:3]-robot[:,0:2], axis=1)
            funcEval = funcEval + np.sum(LA.norm(task[c,1:3]-robot[:,0:2], axis=1))
            print(robot)
    