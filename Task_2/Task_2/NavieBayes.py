import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import math


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
        self.max = np.max(self.X, axis=0)
        for i in range(len(self.X)):
            normX[i] = (self.X[i]-self.mean)/self.std
        return normX

class NB:
    '''
    This class takes input,
        numClasses : int, No. of classes
        numFeatures : int, No. of features
    to initialize the means and standard deviations.
    This algorithm assumes that the data is normally distributed and learns P(x|y) and finds P(y|x) using Bayes theorem.
    '''
    def __init__(self, numClasses, numFeatures):
        self.means = np.zeros((numClasses, numFeatures), dtype=np.float32)
        self.stdev = np.zeros((numClasses, numFeatures), dtype=np.float32)

    def fit(self, X, Y, phi=None):
        '''
        This method is called for training the model. This takes input,
            X : ndarray(m, n), m-No. of samples, n-No. of features
            Y : ndarray(m,)
            phi(optional) : float,
        If phi is not passed or passed as None, it is calculated from the dataset.
        This method just finds the mean and standard deviation of the features of the samples belonging to each class 1 and 0 seperately.
        '''
        self.means[0,:] = np.mean(X[np.where(Y==0)], axis=0)
        self.stdev[0,:] = np.std(X[np.where(Y==0)], axis=0)
        self.means[1,:] = np.mean(X[np.where(Y==1)], axis=0)
        self.stdev[1,:] = np.std(X[np.where(Y==1)], axis=0)
        if not phi:
            self.phi = [1-(len(np.where(Y==1)[0])/len(Y)), len(np.where(Y==1)[0])/len(Y)]
        else:
            self.phi = [1-phi, phi]

    def predict(self, X):
        '''
        This method is called for prediction. It takes input,
            X : ndarray(t, n), t-No. of test samples
        This will find the value of the gaussian function of samples for each feature and class seperately.
        Then it multiplies the probabilities of sample features found using gaussian distribution of each class seperately.
        It then uses Bayes theorem and produces P(y|x) for every sample for both y=0 and y=1.
        It returns the label y for which the above probability is maximum.
        '''
        p = np.zeros((X.shape[0], 2*X.shape[1]), dtype=np.float32)
        
        for i in range(len(X)):
            p[i,:2] = ((X[i,:]-self.means[0,:])**2)/(2*(self.stdev[0,:]**2))
            p[i,2:] = ((X[i,:]-self.means[1,:])**2)/(2*(self.stdev[1,:]**2))
        p = np.exp(-p)
        for i in range(p.shape[1]):
            p[:,i] = p[:,i]*(1/(math.sqrt(2*3.14)*self.stdev[int(i/2),int(i%2)]))
        self.pred = np.zeros((p.shape[0], int(p.shape[1]/2)), dtype=np.float32)
        for i in range(self.pred.shape[1]):
            self.pred[:,i] = p[:,2*i] * p[:,2*i+1]*self.phi[i]
        prediction = np.zeros(self.pred.shape[0])
        for i in range(len(prediction)):
            prediction[i] = (self.pred[i,1]>=self.pred[i,0])
        return prediction
#Data Preprocessing************************************************************************************************************************************************************************

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

#Model Training and predictions************************************************************************************************************************************************************
phi = np.array([None,0.2,0.4,0.6,0.7,0.8])
acc = np.zeros(len(phi), dtype=np.float32)

for j in range(len(phi)):
    model = NB(2, train.X.shape[1])
    model.fit(train.X, train.Y, phi[j])

    pred = model.predict(test.X)
    trainPred = model.predict(train.X)
    trainAcc = 0

    for i in range(len(trainPred)):
        trainAcc += (trainPred[i]==train.Y[i])
    trainAcc /= len(trainPred)
    for i in range(len(pred)):
        acc[j] += (pred[i]==test.Y[i])
    acc[j] /= len(test.Y)

    print(f"The train accuracy of the model for phi={phi[j]} is {trainAcc}")
    print(f"The test accuracy of the model for phi={phi[j]} is {acc[j]}")
print(f"The value of phi for which the model performed the best is {phi[np.where(acc==max(acc))]}\n")

#Sklearn model******************************************************************************************************************************************************************************

skphi = np.array([None,0.2,0.4,0.6,0.7,0.8])
skacc = np.zeros(len(skphi), dtype=np.float32)

for j in range(len(skphi)):
    if skphi[j] is None:
        skmodel = GaussianNB()
    else:
        skmodel = GaussianNB(priors=[1-skphi[j], skphi[j]])
    skmodel.fit(train.X, train.Y)

    skpred = skmodel.predict(test.X)
    sktrainPred = skmodel.predict(train.X)
    sktrainAcc = 0

    for i in range(len(sktrainPred)):
        sktrainAcc += (sktrainPred[i]==train.Y[i])
    sktrainAcc /= len(sktrainPred)
    for i in range(len(skpred)):
        skacc[j] += (skpred[i]==test.Y[i])
    skacc[j] /= len(skpred)

    print(f"The train accuracy of sklearn model for phi={phi[j]} is {sktrainAcc}")
    print(f"The test accuracy of sklearn model for phi={phi[j]} is {skacc[j]}")
print(f"The value of phi for which the sklearn model performed the best is {skphi[np.where(skacc==max(skacc))]}")
