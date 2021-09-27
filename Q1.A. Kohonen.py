#write your code here
#write your code here
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math 

# =========== Scopes =========== #

# Output Space
OutputX, OutputY = 40, 40

# Size
Size = OutputX * OutputY

# =========== Generating Data =========== #

Colors = np.random.rand(Size, 3)

# =========== Initialize Weights =========== #

def Initialize():

    Weights = np.random.ranf((OutputX, OutputY, Colors.shape[1]))
    return Weights

# =========== Find Best Match =========== #
# =========== Competition =========== #

def Winner(Weights):
    SelectedInput = Colors[np.random.choice(Colors.shape[0], size = 1, replace = False)]
    Distances = []

    
    # For All Neurons
    for x in range(OutputX):
        for y in range(OutputY):

            # Calculate Distance Then Select Best
            Distances.append(np.linalg.norm(SelectedInput - Weights[x][y]))

    Minimum = np.min(Distances)

    Map = np.array(Distances).reshape(OutputX, OutputY)
    MinimunArg = np.unravel_index(Map.argmin(), Map.shape)
    
    return Minimum, MinimunArg, Map, SelectedInput

# =========== Cooperation =========== #

def ComputeNeighborhood(MinDistance, Radius):

    PowerOfTwoRadius = 2 * np.pi * pow(Radius, 2)
    
    NeighborX = np.exp(-1 * np.square(np.arange(OutputX) - MinDistance[0]) / PowerOfTwoRadius)
    NeighborY = np.exp(-1 * np.square(np.arange(OutputY) - MinDistance[1]) / PowerOfTwoRadius)

    Neighborhood = np.outer(NeighborX, NeighborY)

    return Neighborhood

# =========== Adaption =========== #
# =========== Updadting Weights =========== #

def UpdateWeights(Weights, LearningRate, Win, Radius, SelectedInput):

    # Neighborhood
    Neighborhood = ComputeNeighborhood(Win, Radius)
    
    iteration = np.nditer(Neighborhood, flags = ['multi_index'])

    # Update Each Neuron
    while not iteration.finished:
        Weights[iteration.multi_index] = Weights[iteration.multi_index] + LearningRate * Neighborhood[iteration.multi_index] * (SelectedInput - Weights[iteration.multi_index])
        iteration.iternext()

# =========== Training =========== #

def Trainging(Iterations, Weights, LearningRate, Radius, Plot):
    for i in range(Iterations):
        Win, WinArg, Map, SelectedInput = Winner(Weights)
        UpdateWeights(Weights, LearningRate, WinArg, Radius, SelectedInput)
        if i % Plot == 0:
            plt.title('After ' + str(i) + ' Iterations')
            plt.imshow(Weights, cmap = "nipy_spectral")
            plt.colorbar()
            plt.show()

# =========== Scopes =========== #

# Radius
Radius = 3

# Learning Rate
LearningRate = 0.6

# Plot
Plot = 60

# Activation Map, Neurons Actually
Map = np.zeros((OutputX, OutputY))

# =========== Train Data =========== #

plt.title('Input Data')
plt.imshow(Colors.reshape(OutputX, OutputY, 3), cmap = "nipy_spectral")
plt.colorbar()
plt.show()

Trainging(200, Initialize(), LearningRate, Radius, Plot)







