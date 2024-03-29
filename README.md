# Multi Robot Task Allocation

Given 6 trucks and 52 cities, can you figure out which truck should go to which city and in what order, so that the team completes the deliveries while spending the least amount of fuel?

This is a very famous combinatorial optimization problem known as the **Multi Agent Travelling Salesman Problem**

Generally a heuristic is hand crafted to decide which truck can be sent to which city.

In this project we ask the question, "Can a neural network be used to learn a good heuristic instead?"

# Problem statement

Given 3 Robots, 6 tasks and current allocation of robot to task, the network should suggest best possible allocation of robots in the next step such that the total distance travelled by the robots is minimised.

# Problem Representation

The task and robot states are given as input to the network.
Each task has a binary ID and the network outputs the ID of the task as an output.

## Task State Representation

### Sample Task Distributions

## Robot State Representation

# Network Architecture

![image](https://user-images.githubusercontent.com/49178343/129748522-0859ba38-6b27-4f92-b316-8d236373a847.png)

# Training - using Particle Swarm Optimization

## Training data
The heuristic learned should be invariant to the distribution of the task locations. Thus the network is trained to minimise the total distance over a wide variety of task distributions. A batch of 1000 randomly distributed task scenarios was used for the same.

## Cost Function

The sum of euclidean distances travelled by all the robots is taken as the cost function to be minimized.

## Constraints

1) Already visited task shouldn't be revisited.
2) More than one robots should not be directed to the same location.
3) Tasks need to be completed within specific number if interations. For this case 2 iterations.

These events are counted and multiplied by scaling factors as costs to be added to the total cost function.

## Optimization scheme

The output of the network does not directly give the cost function, thus a "look up" needs to be performed. This makes the function non differentiable and thus can not be trained by gradient descent based schemes.

Gradient free optimization scheme **Particle Swarm Optimization** is used instead.

PSO is applied with the weights of the network acting as the variables for optimization.

### Hyperparameters
'Num_Networks': 50, 'num_iterations': 500, 'alpha': 0.8, 'BetaLocal': 2, 'BetaGlobal': 2, 'StagnationPenalty': 5, 'InternalIterations': 10

## Weight clipping
The weights of the neural network are clipped to be within -1000 and 1000. This constricts the variable space and helps improve convergence.


# Demo

Network performance before training.

https://user-images.githubusercontent.com/49178343/129745660-6c56bb1d-121f-4602-8382-b36eb13fa242.mp4

Network performance after training

https://user-images.githubusercontent.com/49178343/129745256-cbbbe374-c330-43d6-82b0-7a787de6df75.mp4


# Results

![ConvergenceHistory](https://user-images.githubusercontent.com/49178343/129746891-da8be9ce-2dbb-4d05-96c4-73d4ddf7a991.png)


# Limitations
