# Example Linear Regression

In this example (generalized) linear regression, as introduced [in the previous section](LinReg) is implemented and applied for estimating a function $f()$ that maps the speed of long distance runners to their heartrate. 

$$
heartrate = f(speed)
$$

For training the model, a set of 30 samples is applied, each containing the speed (in m/s) of a runner and the heartrate measured at this speed.  

> Note that in this example input data consists of the single feature *speed*, i.e. it is 1-dimensional (d=1). All functions implemented below are tailored to this one-dimensional case.

Required Modules:

#%matplotlib inline
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

Read data from file. The first column contains an ID, the second column is the speed in m/s and the third column is the heartrate in beats/s.

dataframe=pd.read_csv("HeartRate.csv",header=None,sep=";",decimal=",",index_col=0,names=["speed","heartrate"])
dataframe

numdata=dataframe.shape[0]
print("Number of samples:  ",numdata)

In the function `calculateWeights(X,r,deg)` the weights are calculated by applying the already introduced equation

$$
w=\left( D^T D\right)^{-1} D^T r
$$

The function is tailored to the case, where input data consists of only a single feature. However, the function is implemented such that, it can not only be applied to learn a linear function, but a polynomial of arbitrary degree. The degree of the polynomial can be set by the `deg`-argument of the function.

def calculateWeights(X,r,deg):
    numdata=X.shape[0]
    D=np.zeros((numdata,deg+1))
    for p in range(numdata):
        for ex in range(deg+1):
            D[p][ex]=math.pow(float(X[p]),ex)
    DT=np.transpose(D)
    DTD=np.dot(DT,D)
    y=np.dot(DT,r)
    w=np.linalg.lstsq(DTD,y,rcond=None)[0]
    return w

features=dataframe["speed"].values
targets=dataframe["heartrate"].values

## Learn linear function
First, we learn the best linear function 

$$
heartrate = w_0+w_1 \cdot speed
$$

by setting the `deg`-argument of the function `calculateWeights()` to 1:

degree=1
w=calculateWeights(features,targets,degree)
print('Calculated weights:')
for i in range(len(w)):
    print("w%d = %3.2f"%(i,w[i]))

The learned model and the training samples are plotted below:

plt.figure(figsize=(10,8))
plt.scatter(features,targets,marker='o', color='red')
plt.title('heartrate vs. speed of long distance runners')
plt.xlabel('speed in m/s')
plt.ylabel('heartrate in bps')
RES=0.05 # resolution of speed-axis
# plot calculated linear regression 
minS=np.min(features)
maxS=np.max(features)
speedrange=np.arange(minS,maxS+RES,RES)
hrrange=np.zeros(speedrange.shape[0])
for si,s in enumerate(speedrange):
    hrrange[si]=np.sum([w[d]*s**d for d in range(degree+1)])
plt.plot(speedrange,hrrange)
plt.show()

Finally the mean absolute distance (MAD) and the Mean Square Error (MSE) are calculated.

pred=np.zeros(numdata)
for si,x in enumerate(features):
    pred[si]=np.sum([w[d]*x**d for d in range(degree+1)])
    
mad=1.0/numdata*np.sum(np.abs(pred-targets))
mse=1.0/numdata*np.sum((pred-targets)**2)
print(mad)  
print('MAD = ',mad)   
print('MSE = ',mse)

Note that here the metrics MAD and MSE have been calculated on the training data. Hence, the corresponding values describe how well the model is fitted to training data. But these values are useless for determining how good the model will perform on new data. Usually in Machine Learning performance metrics such as MAD and MSE are calculated on test-data. But in this example we haven't split the set of labeled data into a training- and a test-partition.

## Learn quadratic function

In order to learn the best quadratic function

$$
heartrate = w_0+w_1 \cdot speed +w_2 \cdot (speed)^2
$$

we repeat the steps for `deg=2`:

degree=2
w=calculateWeights(features,targets,degree)
print('Calculated weights:')
for i in range(len(w)):
    print("w%d = %3.2f"%(i,w[i]))

plt.figure(figsize=(10,8))
plt.scatter(features,targets,marker='o', color='red')
plt.title('heartrate vs. speed of long distance runners')
plt.xlabel('speed in m/s')
plt.ylabel('heartrate in bps')
RES=0.05 # resolution of speed-axis
# plot calculated linear regression 
minS=np.min(features)
maxS=np.max(features)
speedrange=np.arange(minS,maxS+RES,RES)
hrrange=np.zeros(speedrange.shape[0])
for si,s in enumerate(speedrange):
    hrrange[si]=np.sum([w[d]*s**d for d in range(degree+1)])
plt.plot(speedrange,hrrange)
plt.show()

pred=np.zeros(numdata)
for si,x in enumerate(features):
    pred[si]=np.sum([w[d]*x**d for d in range(degree+1)])
    
mad=1.0/numdata*np.sum(np.abs(pred-targets))
mse=1.0/numdata*np.sum((pred-targets)**2)
print(mad)  
print('MAD = ',mad)   
print('MSE = ',mse)

## Learn cubic function

In order to learn the best cubic function

$$
heartrate = w_0+w_1 \cdot speed +w_2 \cdot (speed)^2 +w_3 \cdot (speed)^3
$$

we repeat the steps for `deg=3`:

degree=3
w=calculateWeights(features,targets,degree)
print('Calculated weights:')
for i in range(len(w)):
    print("w%d = %3.2f"%(i,w[i]))

plt.figure(figsize=(10,8))
plt.scatter(features,targets,marker='o', color='red')
plt.title('heartrate vs. speed of long distance runners')
plt.xlabel('speed in m/s')
plt.ylabel('heartrate in bps')
RES=0.05 # resolution of speed-axis
# plot calculated linear regression 
minS=np.min(features)
maxS=np.max(features)
speedrange=np.arange(minS,maxS+RES,RES)
hrrange=np.zeros(speedrange.shape[0])
for si,s in enumerate(speedrange):
    hrrange[si]=np.sum([w[d]*s**d for d in range(degree+1)])
plt.plot(speedrange,hrrange)
plt.show()

pred=np.zeros(numdata)
for si,x in enumerate(features):
    pred[si]=np.sum([w[d]*x**d for d in range(degree+1)])
    
mad=1.0/numdata*np.sum(np.abs(pred-targets))
mse=1.0/numdata*np.sum((pred-targets)**2)
print(mad)  
print('MAD = ',mad)   
print('MSE = ',mse)

## Same solution, now using Scikit Learn

degree=3

from sklearn import linear_model
speed=np.transpose(np.atleast_2d(dataframe.values[:,0]))
for d in range(1,degree):
    newcol=np.transpose(np.atleast_2d(np.power(speed[:,0],d+1)))
    speed=np.concatenate((speed,newcol),axis=1)
heartrate=dataframe.values[:,1]

# Train Linear Regression Model
reg=linear_model.LinearRegression()
reg.fit(speed,heartrate)
print(reg)

# Parameters of Trained Model 
print("Degree = ",degree)
print("Learned coefficients w0, w1, w2, ....:")
wlist=[reg.intercept_]
wlist.extend(reg.coef_)
w=np.array(wlist)
print(w)

# Plot training samples
plt.figure(figsize=(10,8))
plt.scatter(speed[:,0],heartrate,marker='o', color='red')
plt.title('heartrate vs. speed of long distance runners')
plt.xlabel('speed in m/s')
plt.ylabel('heartrate in bps')
#plt.hold(True)
for si,s in enumerate(speedrange):
    hrrange[si]=np.sum([w[d]*s**d for d in range(degree+1)])
plt.plot(speedrange,hrrange)
plt.show()

