#!/usr/bin/env python
# coding: utf-8

# # Gaussian Process: Implementation in Python 
# 
# In this section Gaussian Processes regression, as described in the [previous section](gp), is implemented in Python. First the case of predefined mean- and covariance-function is implemented. In the second part these functions are learned from data.

# In[1]:


import numpy as np
from scipy import r_
from matplotlib import pyplot as plt
np.set_printoptions(precision=5,suppress=True)


# ## Gaussian Process for Regression

# Definition of training data. This is the same data as used in the GP regression example in the [previous section](gp).

# In[2]:


xB=np.array([1,2,3,4])
yB=np.array([0.25,0.95,2.3,3.9])


# In[3]:


plt.figure(figsize=(12, 10))
plt.plot(xB,yB,"or",label="Training Data")
plt.legend()
plt.show()


# Define the positions at wich the function values shall be predicted. In contrast to the GP regression in the [previous section](gp), below we predict numeric values not only at the three locations $5,6$ and $7$, but at all locations in the range from 0 to 7 with a resolution of $0.2$: 

# In[4]:


xPred=np.arange(0,7,0.2)


# Define the hyperparameters for the mean- and covariance function

# In[5]:


c2=0.25 # constant coefficient of quadratic term in prior mean function
ell=1 # horizontal length scale parameter in the squared exponential function
sigmaF2=2 #sigmaF2 is the variance of the multivariate gaussian distribution
sigmaN2=0.005 #sigmaN2 is the variance of the regression noise-term


# Definition of mean- and covariance function. Here, the mean function is a quadratic polynomial and the covariance function is the squared exponential.

# In[6]:


def priormean(xin):
    return c2*xin**2

def corrFunc(xa,xb):
    return sigmaF2*np.exp(-((xa-xb)**2)/(2.0*ell**2))


# Calculate the values *mx* of the mean function in the range from 0 to 7. These values are just used for plotting. The values *mxB* are the mean-function values at the training data x-values, i.e. the **mean-vector**. These values are applied for calculating the prediction. 

# In[7]:


mx=priormean(xPred)
mxB=priormean(xB)


# Calculate the covariance matrix by evaluating the covariance function at the training data x-values.

# In[8]:


KB=np.zeros((len(xB),len(xB)))
for i in range(len(xB)):
    for j in range(i,len(xB)):
        noise=(sigmaN2 if i==j else 0)
        k=corrFunc(xB[i],xB[j])+noise
        KB[i][j]=k
        KB[j][i]=k        
print('-'*10+' Matrix KB '+'-'*10)
print(KB.round(decimals=3))


# Calculate the inverse of the covariance matrix

# In[9]:


KBInv=np.linalg.inv(KB)
print('-'*10+' Inverse of Matrix KB '+'-'*10)
print(KBInv.round(decimals=3))


#  Calculate the covariance matrix $K_*$ between training x-values and prediction x-values

# In[10]:


Ks=np.zeros((len(xPred),len(xB)))
for i in range(len(xPred)):
    for j in range(len(xB)):
        k=corrFunc(xPred[i],xB[j])
        Ks[i][j]=k
print('-'*10+' Matrix Ks '+'-'*10)
print(Ks.round(decimals=5))


# Calculate the covariance matrix $K_{**}$ between prediction x-values

# In[11]:


Kss=np.zeros((len(xPred),len(xPred)))
for i in range(len(xPred)):
    for j in range(i,len(xPred)):
        noise=(sigmaN2 if i==j else 0)
        k=corrFunc(xPred[i],xPred[j])+noise
        Kss[i][j]=k
        Kss[j][i]=k
print('-'*10+' Matrix Kss '+'-'*10)
print(Kss.round(decimals=3))


# Calculate the prediction

# In[12]:


mus=priormean(xPred)
ypred=mus+np.dot(np.dot(Ks,KBInv),(yB-mxB))
print("Prediction: ",ypred)


# Calculate the covariance of the predictions

# In[13]:


yvar=np.diag(Kss-np.dot(Ks,np.dot(KBInv,np.transpose(Ks))))
stds=np.sqrt(yvar)
print("Double Standard Deviation: ",2*stds)


# Plot training data and predicitons:

# In[14]:


plt.figure(figsize=(12, 10))
plt.plot(xPred,mx,label="mean $m(x)$")
#plt.hold(True)
plt.plot(xB,yB,'or',label="training data")
plt.plot(xPred,ypred,'--g',label="predictions")
plt.text(0.5,8,"$m(x)=0.25 \cdot x^2$ \n$k(x,x')=2 \cdot \exp(-0.5\cdot(x-x')^2)$ \n $\sigma_n^2=0.005$",fontsize=14)
plt.legend(loc=2,numpoints=1)
plt.title('Gaussian Process Prediction with prior quadratic mean')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis([0,8,0,16])
# plot 2*standard deviation 95%-confidence interval
fillx = r_[xPred, xPred[::-1]]

filly = r_[ypred+2*stds, ypred[::-1]-2*stds[::-1]]
plt.fill(fillx, filly, facecolor='gray', edgecolor='white', alpha=0.3)
plt.show()


# ## Find optimum hyperparameters for mean- and covariance-function

# In[15]:


import numpy as np
from scipy import r_
from matplotlib import pyplot as plt
import scipy.optimize as opt


# In[16]:


xB=np.array([1., 3., 5., 6., 7., 8., 9.])
yB=xB*np.sin(xB)


# In[47]:


def objective(x): #Returns Log-Likelihood, which must be optimized
    mxB=x[0]*xB**2+x[1]*xB+x[2]+x[6]*xB**3+x[7]*xB**4
    KB=np.zeros((len(xB),len(xB)))
    for i in range(len(xB)):
        for j in range(i,len(xB)):
            noise=x[5]**2
            k=x[3]**2*np.exp(-((xB[i]-xB[j])**2)/(2.0*x[4]**2))+noise**2
            KB[i][j]=k
            KB[j][i]=k
    KBinv=np.linalg.inv(KB)
    return -1*(-0.5* np.log(np.linalg.det(KB))-0.5 * np.dot(np.transpose(yB-mxB), np.dot(KBinv,(yB-mxB)))-2*np.log(2*np.pi))


# In[48]:


#Define constraints on the hyperparameters
def constr1(x):
    return x[4]-1 #horizontal length-scale > 1

def constr2(x):
    return 5-x[4] #horizontal length-scale < 5

def constr3(x):
    return x[3]-0.8 #vertical length-scale >0.8


# In[49]:


x0=(0.1, 0.01, 0.01, 2.0, 1.0, 0.01,0.01,0.01) #Startvalues for optimization
xopt=opt.fmin_cobyla(objective,x0,cons=[constr1,constr2,constr3])
print('-'*10+"Results of optimisation"+'-'*10)
print(xopt)


# In[50]:


#####################Definition of hyperparameters#############################
c4=xopt[7] # constant coefficient of biquadratic term in prior mean function
c3=xopt[6] # constant coefficient of cubic term in prior mean function
c2=xopt[0] # constant coefficient of quadratic term in prior mean function
c1=xopt[1] # constant coefficient of linear term in prior mean function
c0=xopt[2] # constant coefficient constant term in prior mean function
ell=xopt[4] # horicontal length scale parameter in the squared exponential function
sigmaF2=xopt[3] #sigmaF2 is the standard deviation of the multivariate gaussian distribution
sigmaN2=xopt[5] #sigmaN2 is the standard deviation of the regression noise-term


# In[51]:


print("Learned mean function m(x) = %1.3f*x^4 + %1.3f*x^3 + %1.3f*x^2+ %1.3f*x + %1.3f"%(c4,c3,c2,c1,c0))
print("Learned cov. function m(x) = (%1.3f)^2 *exp(-(x-x')^2 / (2 * (%1.3f)^2))+ %1.3f^2"%(sigmaF2,ell,sigmaN2))


# In[52]:


###################Definition of mean- and covariance function################# 
def priormean(xin):
    return c4*xin**4+c3*xin**3+c2*xin**2+c1*xin+c0

def corrFunc(xa,xb):
    return sigmaF2**2*np.exp(-((xa-xb)**2)/(2.0*ell**2))


# In[53]:


x=np.arange(0,10,0.1)
mx=priormean(x)
mxB=priormean(xB)


# In[54]:


xPred=np.arange(0,10,0.2)


# In[55]:


KB=np.zeros((len(xB),len(xB)))
for i in range(len(xB)):
    for j in range(i,len(xB)):
        noise=(sigmaN2**2 if i==j else 0)
        k=corrFunc(xB[i],xB[j])+noise**2
        KB[i][j]=k
        KB[j][i]=k        
print('-'*10+' Matrix KB '+'-'*10)
print(KB.round(decimals=3))


# In[56]:


KBInv=np.linalg.inv(KB)
print('-'*10+' Inverse of Matrix KB '+'-'*10)
print(KBInv.round(decimals=3))


# In[57]:


Ks=np.zeros((len(xPred),len(xB)))
for i in range(len(xPred)):
    for j in range(len(xB)):
        k=corrFunc(xPred[i],xB[j])
        Ks[i][j]=k
print('-'*10+' Matrix Ks '+'-'*10)
print(Ks.round(decimals=5))


# In[58]:


Kss=np.zeros((len(xPred),len(xPred)))
for i in range(len(xPred)):
    for j in range(i,len(xPred)):
        noise=(sigmaN2 if i==j else 0)
        k=corrFunc(xPred[i],xPred[j])+noise
        Kss[i][j]=k
        Kss[j][i]=k
print('-'*10+' Matrix Kss '+'-'*10)
print(Kss.round(decimals=3))


# In[59]:


mus=priormean(xPred)
ypred=mus+np.dot(np.dot(Ks,KBInv),(yB-mxB))
print("Prediction: ",ypred)


# In[60]:


yvar=np.diag(Kss-np.dot(Ks,np.dot(KBInv,np.transpose(Ks))))
stds=np.sqrt(yvar)
print("Double Standard Deviation: ",2*stds)


# In[61]:


plt.figure(figsize=(12, 10))
plt.plot(x,mx,label="mean $m(x)$")
#plt.hold(True)
plt.plot(xB,yB,'or',label="training data")
plt.plot(xPred,ypred,'--g',label="predictions")
plt.legend(loc=2,numpoints=1)
plt.title('Gaussian Process Prediction with prior quadratic mean')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis([0,13,-8,10])
# plot 2*standard deviation 95%-confidence interval
fillx = r_[xPred, xPred[::-1]]

filly = r_[ypred+2*stds, ypred[::-1]-2*stds[::-1]]
plt.fill(fillx, filly, facecolor='gray', edgecolor='white', alpha=0.3)
plt.show()


# In[ ]:




