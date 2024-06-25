import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model


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

class LogisticRegression:
    '''
    This Class is the algorithm class.
    This takes input
        numParams : int, No. of features
    to initialize the coefficients 
    '''
    def __init__(self, numParams):
        self.w = np.zeros(numParams, dtype=np.float32)
        self.b = 0

    @staticmethod
    def func(w, x, b):
        '''
        This method takes input
            w : ndarray(n,), coefficients
            x : ndarray(m, n), a sample vector
            b : int, intercept

        It returns ndarray(m,) by computing (1/(1+e**(-w*x+b))) for each sample  
        '''
        z = np.zeros(x.shape[0], dtype=np.float32)
        for i  in range(len(z)):
            z[i] = np.sum(w*x[i,:])+b
        f = 1/(1+np.exp(-z))
        return f

    @staticmethod
    def cost(f, y):
        '''
        This method takes the input,
            f : ndarray(m,), The output of the above method
            y : ndarray(m,), The ground truth or the actual values
        It returns the total cost calculated by the below formula
        '''
        costs = (-y*np.log(f)-((1-y)*np.log(1-f)))
        totalcost = np.sum(costs)
        totalcost /= len(y)
        return totalcost
    
    def fit(self, X, Y, numEpoch, lr):
        '''
        This method is called for training the model.
        This takes the input,
            X : ndarray(m, n),
            Y : ndarray(m,),
            numEpoch : int, No. of training iterations
            lr : float, learning rate
        The below loop updates the parameters 'w' and 'b' using
        w = w - lr*(dj_dw)
        b = b - lr*(dj_db)
        '''
        for epoch in range(numEpoch):
            self.f = self.func(self.w, X, self.b)

            self.totCost = self.cost(self.f, Y)

            dj_db = self.f-Y
            dj_dw = np.zeros(X.shape, dtype=np.float32)
            for i in range(X.shape[1]):
                dj_dw[:,i] = dj_db*X[:,i]
            dj_dw = np.sum(dj_dw, axis=0)/len(Y)
            dj_db = np.sum(dj_db)/len(Y)

            self.w -= lr*dj_dw
            self.b -= lr*dj_db
        print(f"The cost for the learning rate {lr} = {self.totCost}")

    def predict(self, X):
        '''
        This method takes the input
            X : ndarray(t, n), t - No. of test samples
        It returns ndarray(t,) which contains the probabilities of each sample belonging to class y=1
        '''
        pred = self.func(self.w, X, self.b)
        return pred

# Data preprocessing ********************************************************
fileName = "C:\\Users\Skanda\Downloads\ds1_train.csv"
fileName2 = "C:\\Users\Skanda\Downloads\ds1_test.csv"
train = FileReader(fileName)
train.sepXY()

test = FileReader(fileName2)
test.sepXY()

extendedX = np.zeros((train.X.shape[0], train.X.shape[1]+2))
extendedX[:,:2] = train.X
extendedX[:,1] = np.log(train.X[:,1]**2)
extendedX[:,2:] = (train.X**2)

extendedtest = np.zeros((test.X.shape[0], test.X.shape[1]+2))
extendedtest[:,:2] = test.X
extendedtest[:,1] = np.log(test.X[:,1]**2)
extendedtest[:,2:] = (test.X**2)


traindata = DataHandler(extendedX)
normTrainX = traindata.Normalize()

testData = DataHandler(extendedtest)
normTestX = testData.Normalize()
#Training model and predicting for a sat of learning rates****************************************************************************

lr = np.array([0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1])
numLoops = len(lr)
loopCost = np.zeros(len(lr), dtype=np.float32)
loopAcc = np.zeros(len(lr), dtype=np.float32)
loopTrainAcc = np.zeros(len(lr), dtype=np.float32)

for j in range(numLoops):
    model = LogisticRegression(extendedX.shape[1])
    model.fit(normTrainX, train.Y, 100, lr[j])

    loopCost[j] = model.totCost
    pred = model.predict(normTestX)
    trainPred = model.f

    #Since the probabilities are given, it is converted to 1 or 0 using 0.5 as threshold
    pred[np.where(pred>=0.5)]=1
    pred[np.where(pred<0.5)]=0
    trainPred[np.where(trainPred>=0.5)]=1
    trainPred[np.where(trainPred<0.5)]=0
    
    truePred = np.zeros(len(pred))
    truePredTrain=np.zeros(len(trainPred))

    for i in range(len(pred)):
        truePred[i] = (pred[i]==test.Y[i]) #Comparing prediction and ground truth values of test sample for finding accuracy
    for i in range(len(trainPred)):
        truePredTrain[i] = (trainPred[i]==train.Y[i]) #Comparing prediction and ground truth values of training sample for finding accuracy
    
    loopAcc[j] = np.sum(truePred)/len(test.Y)
    loopTrainAcc[j] = np.sum(truePredTrain)/len(train.Y)
    print(f"The test accuracy of the model for learning rate {lr[j]} is {loopAcc[j]}")
    print(f"The train accuracy of the model for learning rate {lr[j]} is {loopTrainAcc[j]}")
print(f"My model coefficients : {model.w, model.b}")
print(f"The learning rate that preformed the best based on test accuracy is {lr[np.where(loopAcc==max(loopAcc))]}")

# Using sklearn model********************************************************************************

skmodel = linear_model.LogisticRegression(penalty=None)
skmodel.fit(normTrainX, train.Y)
print(f"Sklearn coefficients : {skmodel.coef_}, {skmodel.intercept_}")

trainPred = skmodel.predict(normTrainX)
pred = skmodel.predict(normTestX)
truePredsk = np.zeros(len(pred))
trueTrainPredsk = np.zeros(len(trainPred))

for i in range(len(pred)):
    truePredsk[i] = (pred[i]==test.Y[i]) #Comparing prediction and ground truth values of test sample for finding accuracy
for i in range(len(trainPred)): 
    trueTrainPredsk[i] = (trainPred[i]==train.Y[i]) #Comparing prediction and ground truth values of train sample for finding accuracy
    
skTestAcc = np.sum(truePredsk)/len(pred)
skTrainAcc = np.sum(trueTrainPredsk)/len(trainPred)

print(f"Sklearn test accuracy = {skTestAcc}")
print(f"Sklearn train accuracy = {skTrainAcc}")


fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

ax.plot(lr, loopAcc) #Plotting learning rates vs Accuracy
ax1.plot(lr, loopCost) #Plotting learning rates vs Cost 

plt.show()
