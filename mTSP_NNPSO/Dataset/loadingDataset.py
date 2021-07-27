import torch

temp1 = torch.load("Dataset/taskConfig.pt")
print(temp1.shape)
print(temp1[0])