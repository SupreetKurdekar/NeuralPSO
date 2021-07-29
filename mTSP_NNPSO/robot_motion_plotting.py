import numpy as np
import copy
import statistics
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import visual_functions as vf
import utils
import sys
import objectiveFunction
# import the os module
import os

# get relevant model
model_path = "/home/supreet/NeuralPSO/mTSP_NNPSO/results/500_tasks/FinalGlobalBestModel.pth"
model = torch.load(model_path)

# number of task distributions to use
batch_size = 10
StagnationPenalty = 5
InternalIterations = 10

# get task data
taskData = torch.load("mTSP_NNPSO/Dataset/taskConfig.pt")
taskData = taskData.int()
robotData = torch.load("mTSP_NNPSO/Dataset/homeRobotState.pt")
TaskStateBatch = taskData[0:batch_size,:,:].float()
RobotStateBatch = robotData.repeat(batch_size,1,1).float()

case_wise_task_results = []
case_wise_rob_results = []
# get results for batch
functionEvaluation,constraintViolation,robotStateHistory,taskStateHistory = objectiveFunction.objectivefunction_for_visualisation(model,RobotStateBatch,TaskStateBatch,StagnationPenalty,InternalIterations)

# reorder data based on individual task_distributions

for j in range(batch_size):

    task_prog = []
    rob_prog = []
    for i in range(len(taskStateHistory)):
        task_prog.append(taskStateHistory[i][j])
        rob_prog.append(robotStateHistory[i][j])

    task_prog = np.array(task_prog)
    case_wise_task_results.append(task_prog)

    rob_prog = np.array(rob_prog)
    case_wise_rob_results.append(rob_prog)

case_wise_robot_positions = []

import plotly.graph_objects as go

import numpy as np

def plot_one_case(rob_case,task_case):
    robot_1_positions = rob_case[:,0,0:2]
    robot_2_positions = rob_case[:,1,0:2]
    robot_3_positions = rob_case[:,2,0:2]

    task_positions = task_case[0,:,1:3]

    robot_1_positions = np.vstack((np.array([0,0]),robot_1_positions,np.array([0,0])))
    robot_2_positions = np.vstack((np.array([0,0]),robot_2_positions,np.array([0,0])))
    robot_3_positions = np.vstack((np.array([0,0]),robot_3_positions,np.array([0,0])))

    # Create figure
    fig = go.Figure(data=[go.Scatter(x=task_positions[:,0], y=task_positions[:,1],mode="markers",line=dict(width=2, color="blue")),
                        go.Scatter(x=task_positions[:,0], y=task_positions[:,1],mode="markers",line=dict(width=2, color="blue")),
                        go.Scatter(x=task_positions[:,0], y=task_positions[:,1],mode="markers",line=dict(width=2, color="blue")),
                        go.Scatter(x=task_positions[:,0], y=task_positions[:,1],mode="markers",line=dict(width=2, color="blue"))],
            layout=go.Layout(xaxis=dict(range=[-1, 12], autorange=False, zeroline=False),yaxis=dict(range=[-1, 12], autorange=False, zeroline=False),
            title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
            updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None])])]),
        frames=[
go.Frame(data=[go.Scatter(x=[robot_1_positions[k,0]],y=[robot_1_positions[k,1]],mode="markers",marker=dict(color="red", size=10)),
go.Scatter(x=[robot_2_positions[k,0]],y=[robot_2_positions[k,1]],mode="markers",marker=dict(color="green", size=10)),
                    go.Scatter(
                    x=[robot_3_positions[k,0]],
                    y=[robot_3_positions[k,1]],
                    mode="markers",
                    marker=dict(color="cyan", size=10))
                        ])

            for k in range(len(robot_1_positions))]
    )

    return fig

for i in range(batch_size):

    fig = plot_one_case(case_wise_rob_results[i],case_wise_task_results[i])

    fig.show()