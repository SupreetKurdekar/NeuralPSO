import numpy as np
import torch

numRobotSamples = 100
numTaskSamples = 1000

a = np.random.choice(np.arange(1,11), size=(numRobotSamples,3, 3),replace=True)
sampl = np.random.uniform(low=0, high=100, size=(numRobotSamples,3))
a[:,-1,:] = sampl

robotConfig = torch.from_numpy(a)

homeRobotState = np.ones((3,3),dtype=int)
homeRobotState[-1,:] = float(0)
homeRobotState = torch.from_numpy(homeRobotState)
print(homeRobotState)
print(type(homeRobotState))
b = np.random.choice(np.arange(1,11), size=(numTaskSamples,3, 6),replace=True)
b[:,0,:] = 1

p = np.array([[0,0,0,1,1,1],[0,1,1,0,0,1],[1,0,1,0,1,0]])
p = np.repeat(p[np.newaxis,:, :], numTaskSamples, axis=0)

p = np.concatenate((b,p),axis=1)
p = p.astype(float)

print("p",p[0])
print(type(p))
print(p.dtype)
print(p.shape)

taskConfig = torch.from_numpy(p)

print(robotConfig.shape)
print(robotConfig[0])
print(taskConfig.shape)
print(taskConfig[0])

torch.save(homeRobotState,"Dataset/homeRobotState.pt")
torch.save(robotConfig,"Dataset/robotConfig.pt")
torch.save(taskConfig,"Dataset/taskConfig.pt")

