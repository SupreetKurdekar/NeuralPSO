import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def f(x, y,w1,w2):
    # print(1/(1+np.exp(-w1*x)*np.exp(w2*y)))
    return 1/(1+np.exp(-w1*x)*np.exp(-w2*y))

def paraboloid(x,y):
    return x**2 + y**2

class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
    
    def forward(self, x):
        output = self.fc1(x)
        # output = self.sigmoid(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

class Allocator(torch.nn.Module):
    def __init__(self):
        super(Allocator, self).__init__()
        self.convT1 = nn.Conv1d(in_channels=6,out_channels=4,kernel_size=1,stride=1,bias=True)
        self.convT2 = nn.Conv1d(in_channels=4,out_channels=2,kernel_size=1,stride=1,bias=True)

        self.convR1 = nn.Conv1d(in_channels=3,out_channels=2,kernel_size=1,stride=1,bias=True)
        self.convR2 = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=1,stride=1,bias=True)

        self.convM1 = nn.Conv1d(in_channels=2,out_channels=4,kernel_size=2,stride=1,dilation=1,bias=True)  
        self.convM2 = nn.Conv1d(in_channels=2,out_channels=4,kernel_size=2,stride=1,dilation=2,bias=True)
        self.convM3 = nn.Conv1d(in_channels=2,out_channels=4,kernel_size=2,stride=1,dilation=3,bias=True)

        self.ConvF = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(2,6),bias=True)
        
        self.avgPool = nn.AdaptiveMaxPool2d(3)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
        self.relu = torch.nn.ReLU()
    
    def forward(self, x,y):
        
        output1 = self.convT1(x)
        output1 = self.relu(output1)
        output1 = self.convT2(output1)
        output1 = self.relu(output1)

        output2 = self.convR1(y)
        output2 = self.relu(output2)
        output2 = self.convR2(output2)
        output2 = self.relu(output2)

        output = torch.cat((output1,output2),dim=2)

        o1 = self.relu(self.convM1(output))
        o2 = self.relu(self.convM1(output))
        o3 = self.relu(self.convM1(output))
        
        o1 = o1.unsqueeze(dim=1)
        o2 = o2.unsqueeze(dim=1)
        o3 = o3.unsqueeze(dim=1)

        O = torch.cat((o1,o2,o3),dim = 1)

        O = self.sigmoid(self.ConvF(O))
        O1 = O.round()

        return O1

class Allocator2(torch.nn.Module):
    def __init__(self):
        super(Allocator2, self).__init__()
        self.convT1 = nn.Conv1d(in_channels=9,out_channels=6,kernel_size=1,stride=1,bias=True)
        self.convT2 = nn.Conv1d(in_channels=6,out_channels=4,kernel_size=1,stride=1,bias=True)
        self.convT3 = nn.Conv1d(in_channels=4,out_channels=2,kernel_size=1,stride=1,bias=True)

        self.convR1 = nn.Conv1d(in_channels=3,out_channels=2,kernel_size=1,stride=1,bias=True)
        self.convR2 = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=1,stride=1,bias=True)

        self.convM1 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=1,bias=True)  
        self.convM2 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=2,bias=True)
        self.convM3 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=3,bias=True)
        self.convM4 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=4,bias=True)  
        self.convM5 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=5,bias=True)
        self.convM6 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=6,bias=True)
        self.convM7 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=7,bias=True)  
        self.convM8 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=8,bias=True)
        self.convM9 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=9,bias=True)
        self.convM10 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=10,bias=True)  
        self.convM12 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=11,bias=True)
        self.convM13 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=12,bias=True)
        self.convM14 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=13,bias=True)  
        self.convM15 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=14,bias=True)
        self.convM16 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=15,bias=True)
        self.convM17 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=16,bias=True)  
        self.convM18 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=17,bias=True)
        self.convM19 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=18,bias=True)
        self.convM20 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=19,bias=True)  
        self.convM21 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=20,bias=True)
        self.convM22 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=21,bias=True)
        self.convM23 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=22,bias=True)  
        self.convM24 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=23,bias=True)
        self.convM25 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=24,bias=True)
        self.convM26 = nn.Conv1d(in_channels=2,out_channels=7,kernel_size=2,stride=1,dilation=25,bias=True)
        # self.transformer = nn.

        self.ConvF1 = nn.Conv2d(in_channels=25,out_channels=16,kernel_size=(2,6),bias=True)
        self.ConvF2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=(2,6),bias=True)
        self.ConvF3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=(2,6),bias=True)
        
        self.avgPool = nn.AdaptiveMaxPool2d(3)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
        self.relu = torch.nn.ReLU()
    
    def forward(self, x,y):
        output1 = self.convT1(x)
        output1 = self.relu(output1)
        output1 = self.convT2(output1)
        output1 = self.relu(output1)
        output1 = self.convT3(output1)

        output2 = self.convR1(y)
        output2 = self.relu(output2)
        output2 = self.convR2(output2)
        output2 = self.relu(output2)

        output = torch.cat((output1,output2),dim=2)

        o1 = self.relu(self.convM1(output))
        o2 = self.relu(self.convM2(output))
        o3 = self.relu(self.convM3(output))
        o4 = self.relu(self.convM4(output))
        o5 = self.relu(self.convM5(output))
        o6 = self.relu(self.convM6(output))
        o7 = self.relu(self.convM7(output))
        o8 = self.relu(self.convM8(output))
        o9 = self.relu(self.convM9(output))
        o10 = self.relu(self.convM10(output))
        o12 = self.relu(self.convM12(output))
        o13 = self.relu(self.convM13(output))
        o14 = self.relu(self.convM14(output))
        o15 = self.relu(self.convM15(output))
        o16 = self.relu(self.convM16(output))
        o17 = self.relu(self.convM17(output))
        o18 = self.relu(self.convM18(output))
        o19 = self.relu(self.convM19(output))
        o20 = self.relu(self.convM20(output))
        o21 = self.relu(self.convM21(output))
        o22 = self.relu(self.convM22(output))
        o23 = self.relu(self.convM23(output))
        o24 = self.relu(self.convM24(output))
        o25 = self.relu(self.convM25(output))
        o26 = self.relu(self.convM26(output))
        
        o1 = o1.unsqueeze(dim=1)
        o2 = o2.unsqueeze(dim=1)
        o3 = o3.unsqueeze(dim=1)
        o4 = o4.unsqueeze(dim=1)
        o5 = o5.unsqueeze(dim=1)
        o6 = o6.unsqueeze(dim=1)
        o7 = o7.unsqueeze(dim=1)
        o8 = o8.unsqueeze(dim=1)
        o9 = o9.unsqueeze(dim=1)
        o10 = o10.unsqueeze(dim=1)
        o12 = o12.unsqueeze(dim=1)
        o13 = o13.unsqueeze(dim=1)
        o14 = o14.unsqueeze(dim=1)
        o15 = o15.unsqueeze(dim=1)
        o16 = o16.unsqueeze(dim=1)
        o17 = o17.unsqueeze(dim=1)
        o18 = o18.unsqueeze(dim=1)
        o19 = o19.unsqueeze(dim=1)
        o20 = o20.unsqueeze(dim=1)
        o21 = o21.unsqueeze(dim=1)
        o22 = o22.unsqueeze(dim=1)
        o23 = o23.unsqueeze(dim=1)
        o24 = o24.unsqueeze(dim=1)
        o25 = o25.unsqueeze(dim=1)
        o26 = o26.unsqueeze(dim=1)
        
        O = torch.cat((o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o12,o13,o14,o15,o16,o17,o18,o19,o20,o21,o22,o23,o24,o25,o26),dim = 1)

        O = self.sigmoid(self.ConvF1(O))
        O = self.sigmoid(self.ConvF2(O))
        O = self.sigmoid(self.ConvF3(O))
        O1 = O.round()

        return O1



