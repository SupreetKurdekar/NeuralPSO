import numpy as np
import torch
import torch.nn as nn
import copy

def WeightListToVector(listOfNetworkWeights):
    listOfWeightsForAllNetworks = []
    for network in listOfNetworkWeights:
        networkVectorList = []
        for weight in network:
            if len(networkVectorList) == 0:
                networkVectorList.append(weight.flatten())
            else:
                np.r_[networkVectorList[0],weight.flatten()]
        listOfWeightsForAllNetworks.append(networkVectorList[0])

    return np.array(listOfWeightsForAllNetworks)

def state_dict_from_vector(listOfWeightShapes,splits,positions,networks):
    for position,network in zip(positions,networks):
        splitVectors = np.hsplit(position,splits)
        for splitVector,shape in zip(splitVectors,listOfWeightShapes):
            tensor = torch.from_numpy(splitVector.reshape(shape))
    
    return 1

# objectiveFunction()
def vec_bin_array(arr, m):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret 


def weightListFromModel(model):
    layerList = []
    layerShapeList = []
    for name, param in model.named_parameters():
        # print(name, param.size())
        layerList.append(param.data)
        layerShapeList.append(param.size())
    return layerList,layerShapeList

def modelToWeightList(model):
    layerList,layerShapeList = weightListFromModel(model)

    # print(layerShapeList)
    # print("init tensor",layerList[2])
    flattenedWeightList = [i.reshape(-1) for i in layerList]

    # for i in flattenedWeightList:
    #     print(i.shape)

    # print("tensor",layerList[2])
    # print("reshaped tensor",flattenedWeightList[2])

    numpyWeightList = [i.numpy() for i in flattenedWeightList]
    # print("numpy tensor",numpyWeightList[2])

    modelWeightVector = np.concatenate(numpyWeightList, axis=0 )

    variable = [i for i in modelWeightVector]
    # print(variable)
    return variable,layerShapeList

def updateModelWithNewWeightList(weightList,model):
    id = 0
    state = copy.deepcopy(model.state_dict())
    # print(state)

    for name in state:
        numElem = state[name].numel()
        shape = state[name].size()
        subArray = np.array(weightList[id:id+numElem],dtype='float32')
        subArray = subArray.reshape(shape)
        subArray = torch.from_numpy(subArray)
        id = id+numElem
        state[name] = subArray

    model.load_state_dict(state)
    # for name, param in model.named_parameters():
    #     numElements = param.numel()
    #     shape = param.size()
    #     subArray = np.array(weightList[id:id+numElements])
    #     subArray = subArray.reshape(shape)
    #     subArray = torch.from_numpy(subArray)
    #     id = id+numElements
    #     param.data = subArray

    return model