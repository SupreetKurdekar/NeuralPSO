import numpy as np

def Dejong(population):
    population = population**2
    answer = np.sum(population,axis = 1)

    return np.expand_dims(answer,axis=1)

def Rosenbrock(positions):
    squaredPositions = positions**2
    A = 100*np.sum((positions[:,1:] - squaredPositions[:,:-1])**2,axis=1)
    B = np.sum(1-positions[:,:-1])

    C = A + B

    return np.expand_dims(C,axis=1)

def Rastrigin(positions):
    squaredPositions = positions**2
    ans = 10*positions.shape[1] + np.sum(squaredPositions - 10*np.cos(2*np.pi*positions),axis=1)

    return np.expand_dims(ans,axis=1)

def Griewanks(positions):

    rootI = np.sqrt(np.arange(positions.shape[1])+1)
    dividedPositions = positions/rootI
    a = np.sum(positions**2,axis=1)/4000
    b = np.prod(np.cos(dividedPositions))

    return np.expand_dims(a-b+1,axis=1)

def schwefel(positions):

    ans = -np.sum(positions*np.sin(np.sqrt(np.abs(positions))),axis=1)

    return np.expand_dims(ans,axis=1)

def ackleys(positions,a=20,b=0.2,c=2*np.pi):

    n = positions.shape[1]
    term1 = -a*np.exp(-b*np.sqrt(np.sum(positions**2,axis=1)/n))
    term2 = -np.exp(np.sum(np.cos(c*positions),axis=1)/n)

    ans = term1 + term2 + a + np.exp(1)

    return np.expand_dims(ans,axis=1)

def Michaelwicz(positions,m):
    
    i = np.arange(positions.shape[1])+1

    ans = -np.sum(np.sin(positions)*((np.sin(i*(positions**2)/np.pi))**(2*m)),axis=1)

    return np.expand_dims(ans,axis=1)

def constraintViolation(positions):
    return np.zeros((len(positions),1))