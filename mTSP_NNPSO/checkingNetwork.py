import torch
import numpy as np
import objectiveFunction

model = torch.load("D:\\NNPSO_Results\\Run31\\InterimModel260.pth")
taskData = torch.load("Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:2,:,:].float()
RobotStateBatch = robotData.repeat(2,1,1).float()
func,conv = objectiveFunction.objectivefunction(model,RobotStateBatch,TaskStateBatch,5,10)
func1,conv1 = objectiveFunction.TaskStateDisplay(model,RobotStateBatch,TaskStateBatch,5,10)

print("func eval",func,"Constraint Violation",conv)