#write your code here

# Use both of the codes above and plot all of them together
# RBF, MLP with the real function for f(x) = sin(x)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# ~~~~~~~~~~~~~~~~~ MLP ~~~~~~~~~~~~~~~~~~~~~ #

# =========== Keras Model =========== #

Model = Sequential()
Model.add(Dense(40, input_shape=(1,), activation = 'sigmoid'))
Model.add(Dense(20, input_shape=(1,), activation = 'sigmoid'))
Model.add(Dense(1))

# =========== Complie & Run =========== #

# Model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["acc"])
Model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

# ~~~~~~~~~~~~~~~~~ MLP ~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~ RBF ~~~~~~~~~~~~~~~~~~~~~ #

# =========== Utils =========== #

def Gaussian(Input, Mean, Deviation):
    return np.exp(-1 * pow(Input - Mean, 2) / (2 * pow(Deviation, 2)))

def KClustering(X, k):
    # Randomly Select Initial Clusters From Input Data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # Find The Cluster That's Closest To Each Point
        closestCluster = np.argmin(distances, axis=1)

        # Update Clusters By Taking The Mean Of All Of The Points Assigned To That Cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds

# =========== RBF Model =========== #

class RBF(object):

    def __init__(self, K = 2, LearningRate = 0.1, Epochs = 100, Function = Gaussian, InferDeviations = True):
        self.K = K
        self.LearningRate = LearningRate
        self.Function = Function
        self.Epochs = Epochs
        self.InferDeviations = InferDeviations

        # We Only Have One OUtput Neuron
        # So Need One Bias, And A 1D Vector For Weights
        self.Weights = np.random.randn(K)
        self.Bias = np.random.randn(1)

    def Fit(self, Input, Y):

        if self.InferDeviations == True:
            self.Centers, self.Deviations = KClustering(Input, self.K)
        else:
            # Use A Fixed Deviation
            self.Centers, _ = KClustering(Input, self.K)
            MaxD = max([np.abs(c1 - c2) for c1 in self.Centers for c2 in self.Centers])
            self.Deviations = np.repeat(MaxD / np.sqrt(2 * self.K), self.K)

        for e in range(self.Epochs):

            # We Do This For Each Input
            for x in range(Input.shape[0]):

                #For All Of The Radial Basis Functions
                EachInputResult = np.array([self.Function(Input[x], Center, Deviation) for Center, Deviation in zip(self.Centers, self.Deviations)])
                Result = np.dot(EachInputResult.T, self.Weights) + self.Bias

                # Calculate Error
                Error = Y[x] - Result

                # Update
                self.Weights = self.Weights + self.LearningRate * Error * EachInputResult
                self.Bias = self.Bias + self.LearningRate * Error


    def Predict(self, Input):
        Predicted = []
        for x in range(Input.shape[0]):

            #For All Of The Radial Basis Functions
            EachInputResult = np.array([self.Function(Input[x], Center, Deviation) for Center, Deviation in zip(self.Centers, self.Deviations)])
            Result = np.dot(EachInputResult.T, self.Weights) + self.Bias
            Predicted.append(Result)
        return Predicted

# ~~~~~~~~~~~~~~~~~ RBF ~~~~~~~~~~~~~~~~~~~~~ #


# =========== Train Data RBF =========== #


x = np.arange(-400, 400) / 100
y = np.sin(x)

RBFModel = RBF(LearningRate = 0.001, K = 2)
RBFModel.Fit(x, y)

# =========== Summary MLP =========== #

print(Model.summary())
Model.fit(x, y, epochs = 1000, batch_size = 8)

# =========== Test And Show =========== #

YPredictedRBF = RBFModel.Predict(x)
PredsMLP = Model.predict(x)
 

plt.ylabel('Y Vs Predicted Value')
plt.xlabel('X Value')

plt.plot(x, y, 'bo', label = 'Sin(x)')
plt.plot(x, YPredictedRBF, 'g^', label = 'RBF')
plt.plot(x, PredsMLP, 'r^', label = 'MLP')

plt.legend()
 
plt.tight_layout()
plt.show()

