import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FileReader:
    '''
    This class is used to read csv file.
    The sepXY method seperates the sample features 'X' and labels 'Y'
    '''
    def __init__(self, fileName : str)->None:
        self.file = open(fileName, 'r')
        self.csvFile = csv.reader(self.file)

    def sepXY(self):
        self.data = []
        for data in self.csvFile:
            self.data.append(data)
        self.data = self.data[1:]
        self.data = np.array(self.data, dtype=np.float32)
        
        self.X = self.data[:, :2]
        self.Y = self.data[:, 2]

class DataHandler:
    '''
    This class is for normalizing the data that will help the algorithm perform better
    This takes input
        X : ndarray(m, n), m - No. of sample, n - No. of features
    Normalizing can be done by the method I have used below and also by dividing the sample by their maximum values.
    If the sample values after normalizing using the below process gives extremely large values during computation,
    we can divide them by their maximum values to get all the values between 0 and 1.
    '''
    def __init__(self, X):
        self.X = X

    def Normalize(self):
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        normX = np.zeros(self.X.shape)
        self.maxX = np.max(self.X, axis=0)
        for i in range(len(self.X)):
            normX[i] = (self.X[i]-self.mean)/self.std
        return normX

class GDA:
    '''
    This Class is the algorithm class.
    This takes input
        numClasses : int, No. of classes
        numFeatures : int, No. of features
    to initialize the means
    '''
    def __init__(self, numClasses, numFeatures):
        self.numClasses = numClasses
        self.means = np.zeros((numClasses, numFeatures), dtype=np.float32)

    def fit(self, X, Y, phi=None):
        '''
        This method is called for training the model.
        This takes the input,
            X : ndarray(m, n),
            Y : ndarray(m,),
            phi(optional) : float, class prior 
        If phi is not passed or passed as None, it is calcuated from the data.
        Below, the means of each class, covariance matrix(sigma) and phi(if None) are calculated assuming the samples are normally distributed.
        '''
        for i in range(self.numClasses):
            self.means[i,:] = np.sum(X[np.where(Y==i)], axis=0)
        for i in range(self.numClasses):
            self.means[i,:] /= len(np.where(Y==i)[0])
        if not phi:
            self.phi = np.sum(Y)/len(Y) # P(y=1)
        else:
            self.phi = phi
        self.sigma = np.zeros((self.means.shape[1], self.means.shape[1]), dtype=np.float32)
        for i in range(X.shape[0]):
            self.sigma += np.matmul(np.transpose([X[i,:]-self.means[int(Y[i]),:]]), ([X[i,:]-self.means[int(Y[i]),:]]))
        self.sigma /= X.shape[0]
    def predict(self, X):
        '''
        This method is called for prediction.
        This takes input,
            X : ndarray(t, n), t - No. of test samples
        It returns ndarray(No. of classes, t), where element (i,j) contains the probability of 'j'th sample belonging to class 'i' claculated using Bayes' theorem.
        '''
        e = np.zeros((2, X.shape[0]), dtype=np.float32)
        for i in range(X.shape[0]):
            t0 = np.matmul(([X[i,:]-self.means[0]]), np.linalg.inv(self.sigma))
            t1 = np.matmul(([X[i,:]-self.means[1]]), np.linalg.inv(self.sigma))
            for j in range(X.shape[1]):
                e[0,i] = e[0,i] + t0[0,j]*((X[i,:]-self.means[0])[j])
                e[1,i] = e[1,i] + t1[0,j]*((X[i,:]-self.means[1])[j])
        p = (1/((2*3.14)**(X.shape[1]/2))*np.linalg.det(self.sigma))*(np.exp(-e/2))
        p[0,:] *= (1-self.phi)
        p[1,:] *= self.phi
        return p

#Data Preprocessing***************************************************************************************
fileName = "C:\\Users\Skanda\Downloads\ds1_train.csv"
fileName2 = "C:\\Users\Skanda\Downloads\ds1_test.csv"

train = FileReader(fileName)
train.sepXY()

test = FileReader(fileName2)
test.sepXY()

trainData = DataHandler(train.X)
testData = DataHandler(test.X)

normTrain = trainData.Normalize()
normTest = testData.Normalize()
#Training Model and Prediction****************************************************************************************************************
phi = np.array([None, 0.2, 0.4, 0.6, 0.8])
numLoops = len(phi)
loopAcc = np.zeros(numLoops, dtype=np.float32)
loopTrainAcc = np.zeros(numLoops, dtype=np.float32)

for j in range(numLoops):
    model = GDA(2, train.X.shape[1])
    model.fit(normTrain, train.Y, phi[j])

    trainPred = model.predict(normTrain)
    pred = model.predict(normTest)
    truePred = np.zeros(pred.shape[1])
    truePredTrain = np.zeros(trainPred.shape[1])
    for i in range(pred.shape[1]):
        if (pred[0,i]>pred[1,i]):
            truePred[i] = 0
        else:
            truePred[i] = 1
    for i in range(len(truePredTrain)):
        truePredTrain[i] = (trainPred[1,i]>=trainPred[0,i])
    for i in range(len(truePred)):
       loopAcc[j] += (truePred[i]==test.Y[i])
    for i in range(len(truePredTrain)):
        loopTrainAcc[j] += (truePredTrain[i]==train.Y[i])
    loopAcc[j] /= len(test.Y)
    loopTrainAcc[j] /= len(train.Y)
    print(f"The train accuracy for phi = {phi[j]} is {loopTrainAcc[j]}")
    print(f"The test accuracy for phi = {phi[j]} is {loopAcc[j]}")
print(f"The phi for which the model performed best based on test accuracy is {phi[np.where(loopAcc==max(loopAcc))]}\n")

#Using sklearn model ******************************************************************************************

skphi = np.array([[0,0], [0.8,0.2], [0.6,0.4], [0.4,0.6], [0.2,0.8]])
skloopAcc = np.zeros(numLoops, dtype=np.float32)
skloopTrainAcc = np.zeros(numLoops, dtype=np.float32)

for j in range(numLoops):
    if (j==0):
        skmodel = LinearDiscriminantAnalysis()
    else:
        skmodel = LinearDiscriminantAnalysis(priors=skphi[j])
    skmodel.fit(normTrain, train.Y)

    sktrainPred = skmodel.predict(normTrain)
    skpred = skmodel.predict(normTest)

    for i in range(len(skpred)):
        skloopAcc[j] += (skpred[i]==test.Y[i])
    for i in range(len(sktrainPred)):
        skloopTrainAcc[j] += (sktrainPred[i]==train.Y[i])
    skloopAcc[j] /= len(test.Y)
    skloopTrainAcc[j] /= len(train.Y)

    print(f"The training accuracy of sklearn model for phi = {skphi[j,1]} is {skloopTrainAcc[j]}")
    print(f"The test accuracy of sklearn model for phi = {skphi[j,1]} is {skloopAcc[j]}")
print(f"The phi for which the sklearn model performed best based on test accuracy is {phi[np.where(skloopAcc==max(skloopAcc))]}")
fig, ax = plt.subplots()
ax.plot(phi, loopAcc)
ax.plot(phi, loopTrainAcc)
plt.show()

