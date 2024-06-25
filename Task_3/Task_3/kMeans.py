import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

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

class kMeans:
    '''
    This Class is the algorithm class.
    This takes input
        numClasses : int, No. of classes
        numFeatures : int, No. of features to initialize the means
        numLoops : int, No. of times kMeans Algorithm is run with random initializations of means before returning final result to get more optimized results.
    '''
    def __init__(self, numClasses, numFeatures, numLoops):
        self.numClasses = numClasses
        self.numFeatures = numFeatures
        self.means = np.zeros((numClasses, numFeatures), dtype=np.float32)
        self.numLoops = numLoops

    def fit(self, X, numEpoch):
        '''
        This method is called for training the model.
        This takes the input,
            X : ndarray(m, n),
            Y : ndarray(m,),
            numEpoch : int, No. of iterations of the kMeans Algorithm
        In every loop, first the labels are assigned to the samples according to the cluster centers that are closest to them.
        Then the average of the samples that belong to a particular cluster are taken and the means are updated.
        '''
        
        self.dist = np.zeros((self.numClasses, X.shape[0]), dtype=np.float32)
        self.c = np.zeros(X.shape[0]) #Contains the labels assigned during training
        self.tempMean = np.zeros((2*self.numLoops, self.numFeatures), dtype=np.float32) #To contain the means for every iteration before choosing the best one
        self.cost = np.zeros(self.numLoops, dtype=np.float32) #To contain the accuracy of each iteration
        
        for numLoop in range(self.numLoops):
            self.tempMean[2*numLoop,:] = X[random.randint(0,X.shape[0]-1),:]
            self.tempMean[2*numLoop+1,:] = X[random.randint(0,X.shape[0]-1),:]
            for epoch in range(numEpoch):
                for i in range(X.shape[0]):
                    self.dist[0,i] = (np.sum((X[i,:]-self.tempMean[2*numLoop,:])**2))
                    self.dist[1,i] = (np.sum((X[i,:]-self.tempMean[2*numLoop+1,:])**2))
                for i in range(X.shape[0]):
                    self.c[i] = (self.dist[1,i]<self.dist[0,i])
                self.tempMean[2*numLoop,:] = np.sum(X[np.where(self.c==0)], axis=0)/len(np.where(self.c==0)[0])
                self.tempMean[2*numLoop+1,:] = np.sum(X[np.where(self.c==1)], axis=0)/len(np.where(self.c==1)[0])
            for i in range(X.shape[0]):
                self.cost[numLoop] += self.dist[int(self.c[i]),i]

        #print((np.where(self.acc==max(self.acc))[0]))
        self.means[0,:] = self.tempMean[2*(np.where(self.cost==min(self.cost))[0][0]),:]
        self.means[1,:] = self.tempMean[2*(np.where(self.cost==min(self.cost))[0][0])+1,:]

    def predict(self, X):
        '''
        This method is used for prediction.
        This takes input,
            X : ndarray(t, n), t-No. of test samples
        It returns ndarray(t,) with the labels of clusters whose centers are closest to them. 
        '''
        self.k = np.zeros(X.shape[0])
        self.dist = np.zeros((self.numClasses, X.shape[0]), dtype=np.float32)
        for i in range(X.shape[0]):
            self.dist[0,i] = np.sum((X[i,:]-self.means[0,:])**2)
            self.dist[1,i] = np.sum((X[i,:]-self.means[1,:])**2)
        for i in range(X.shape[0]):
            self.k[i] = (self.dist[1,i]<self.dist[0,i])
        return self.k
#Data Preprocessing**********************************************************************************************************
    
fileName = "C:\\Users\Skanda\Downloads\ds1_train.csv"
fileName2 = "C:\\Users\Skanda\Downloads\ds1_test.csv"
train = FileReader(fileName)
train.sepXY()

test = FileReader(fileName2)
test.sepXY()

trainData = DataHandler(train.X)
normTrain = trainData.Normalize()
testData = DataHandler(test.X)
normTest = testData.Normalize()

#Model training and prediction***********************************************************************************************
nLoops = [1,2,3,4,5,10,30]
niterations = [5,10,20,30,50,100]
acc = np.zeros((len(nLoops), len(niterations)), dtype=np.float32)
for j in range(len(nLoops)):
    for k in range(len(niterations)): 
        model = kMeans(2,2,nLoops[j])
        model.fit(normTrain, niterations[k])
#print((model.acc))
        pred = model.predict(normTest)
        trainAcc=0
        acc[j,k]=0
        for i in range(len(pred)):
            acc[j,k] += (pred[i]==test.Y[i])
        acc[j,k] /= len(pred)

        for i in range(len(model.c)):
            trainAcc += (model.c[i]==train.Y[i])
        trainAcc /= len(model.c)
        if (trainAcc<0.5):
            trainAcc = 1-trainAcc
        if (acc[j,k]<0.5):
            pred = (1-pred)
            acc[j,k] = 1-acc[j,k]
        print(f"The train accuracy of my model for [{nLoops[j]},{niterations[k]}] is {trainAcc}")
        print(f"The test accuracy of my model for [{nLoops[j]},{niterations[k]}] is {acc[j,k]}")
acc = np.reshape(acc,(1,-1))
bestInd = np.where(acc[0,:]==max(acc[0,:]))[0][0]
print(f"The combination for which the model performed the best is {[nLoops[int(bestInd/len(niterations))], niterations[int(bestInd % len(niterations))]]}")

#Model using scikit learn******************************************************************************************************

skmodel = KMeans(n_clusters=2, n_init=50)
skmodel.fit(normTrain)
skpred = skmodel.predict(normTest)
sktrainPred = skmodel.predict(normTrain)
skacc=0
skTrainAcc = 0
for i in range(len(skpred)):
    skacc += (skpred[i]==test.Y[i])

for i in range(len(sktrainPred)):
    skTrainAcc += (sktrainPred[i]==train.Y[i])

skTrainAcc /= len(sktrainPred)
skacc /= len(skpred)

if (skTrainAcc<0.5):
    skTrainAcc = (1-skTrainAcc)
    
if (skacc<0.5):
    skpred = (1-skpred)
    skacc = 1-skacc

print(f"The train accuracy of scikit algorithm is {skTrainAcc}")
print(f"The test accuracy of scikit algorithm is {skacc}")
