# Bayes- and Naive Bayes Classifier

> In this section and nearly all other parts of this course basic notions of **probability theory** are required. If you feel unsure about this, it is strongly recommended to study this [short intro in probability theory](https://hannibunny.github.io/probability/intro.html).  

In this notebook a Bayesian classifier for 1-dimensional input data is developed. The task is to predict the **category of car ($C_i$)**, a customers will purchase, if his **annual income ($x$)** is known. 

The classification shall be realized by applying Bayes-Theorem, which in this context is:

$$
P(C_i|x)=\frac{p(x|C_i)P(C_i)}{P(x)} = \frac{p(x|C_i)P(C_i)}{\sum_k p(x|C_k)P(C_k)}
$$

In the **training phase** the gaussian distributed likelihood $p(x|C_i)$ and the a-priori $P(C_i)$ for each of the 3 car classes $C_i$ is estimated from a sample of 27 training instances, each containing the annual income and the purchased car of a former customer. The file containing the training data can be ob obtained from [here](AutoKunden.txt)    

Required Python modules:

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=5,suppress=True)

## Access labeled data
Read customer data from file. Each row in the file represents one custoner. The first column is the customer ID, the second column is the annual income of the customer and the third column is the class of car he or she bought: 

* 0 = Low Class
* 1 = Middle Class
* 2 = Premium Class

autoDF=pd.read_csv("AutoKunden.csv",index_col=0)#,header=None,names=["income","class"],sep="  ",index_col=0)

autoDF

The above data shall be applied for training the classifier. **The trained model shall then be applied to classify customers, whose annual income is defined in the list below:**

AnnualIncomeList=[25000,29000,63000,69000] #customers with this annual income shall be classified

## Training
In the training-phase for each car-class $C_i$ the likelihood-function $p(x|C_i)$ and the a-priori probability $p(C_i)$ must be determined. It is assumed that the likelihoods are gaussian normal distributions. Hence, for each class the **mean** and the **standard-deviation** must be estimated from the given data. 

classStats=autoDF.groupby(by="class").agg({"class":"count","income":["mean","std"]})
classStats["apriori"]=classStats["class","count"].apply(lambda x:x/autoDF.shape[0])
classStats

plt.figure(figsize=(10,8))
Aposteriori=[]
x=list(range(0,100000,100))
for c in classStats.index:
    p=classStats["apriori"].values[c]
    m=classStats["income"]["mean"].values[c]
    s=classStats["income"]["std"].values[c]
    likelihood = 1.0/(s * np.sqrt(2 * np.pi))*np.exp( - (x - m)**2 / (2 * s**2) )
    aposterioriMod=p*likelihood
    Aposteriori.append(aposterioriMod)
    plt.plot(x,aposterioriMod,label='class '+str(c))
plt.grid(True)
for AnnualIncome in AnnualIncomeList: #plot vertical lines at the annual incomes for which classification is required
    plt.axvline(x=AnnualIncome,color='m',ls='dashed')
plt.legend()
plt.xlabel("Annual Income")
plt.ylabel("Probability")
plt.title("Likelihood times A-Priori Probability for all 3 classes")
plt.show()

## Classification (Inference Phase)

Once the model is trained the likelihood $p(x|C_i)$ and the a-priori probability $P(C_i)$ is known for all 3 classes $C_i$. 

The most probable class is then calculated as follows: 

$$
C_{pred} = argmax_{C_i}\left( \frac{p(x|C_i) \cdot p(C_i)}{p(x)}\right) = argmax_{C_i}\left( \frac{p(x|C_i)P(C_i)}{\sum_k p(x|C_k)P(C_k)}\right) 
$$

In the code-cell below, customers with incomes of $25.000.-,29000.-,63000.-$ and $69000.-$ Euro are classified by the learned model:

for AnnualIncome in AnnualIncomeList:
    print('-'*20)
    print("Annual Income = %7.2f"%AnnualIncome)
    i=int(round(AnnualIncome/100))
    proVal=[x[i] for x in Aposteriori]
    sumProbs=np.sum(proVal)
    for i,p in enumerate(proVal):
        print('APosteriori propabilitiy of class %d = %1.4f'% (i,p/sumProbs))
    print('Most probable class for customer with income %5.2f Euro is %d '% (AnnualIncome,np.argmax(np.array(proVal))))

## Bayesian Classification with Scikit-Learn
For Bayesian Classification Scikit-Learn provides Naive Bayes Classifiers for Gaussian-, Bernoulli- and Multinomial distributed data. In the example above 1-dimensional Gaussian distributed input-data has been applied. In this case the Scikit-Learn Naive Bayes Classifier for Gaussian-distributed data, `GaussianNB` learns the same model as the classifier implemented in the previous sections of this notebook. This is demonstrated in the following code-cells:

from sklearn.naive_bayes import GaussianNB

Income = np.atleast_2d(autoDF.values[:,0]).T
labels = autoDF.values[:,1]

### Train the Naive Bayes Classifier:

clf=GaussianNB()
clf.fit(Income,labels)

The parameters mean and standarddeviation of the learned likelihoods are:

print("Learned mean values for each of the 3 classes: \n",clf.theta_)
print("Learned standard deviations for each of the 3 classes: \n",np.sqrt(clf.sigma_))
print("Note that std is slightly different as above. This is because std of pandas divides by (N-1)")

### Use the trained model for predictions

Income=np.atleast_2d(AnnualIncomeList).T
predictions=clf.predict(Income)
for inc,pre in zip(AnnualIncomeList,predictions):
    print("Most probable class for annual income of %7d.-Euro is %2d"%(inc,pre))

The `predict(input)`-method returns the estimated class for the given input. If the a-posteriori probability $P(C_i|\mathbf{x})$ is of interest, the `predict_proba(input)`-method can be applied:

predictionsProb=clf.predict_proba(Income)
for i,inc in enumerate(AnnualIncomeList):
    print("A-Posteriori for class 0: %1.3f ; class 1: %1.3f ; class 3 %1.3f for user with income %7d"%(predictionsProb[i,0], predictionsProb[i,1],predictionsProb[i,2],inc))

### Model Accuracy on training data

Income=np.atleast_2d(autoDF.values[:,0]).T
predictions=clf.predict(Income)

correctClassification=predictions==labels
print(correctClassification)

numCorrect=np.sum(correctClassification)

accuracyTrain=float(numCorrect)/autoDF.shape[0]
print("Accuracy on training data is: %1.3f"%accuracyTrain)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=labels,y_pred=predictions)

In the confusion matrix the entry $C_{i,j}$ in row $i$, column $j$ is the number of instances, which are known to be in class $i$, but predicted to class $j$. For example the confusion matrix above indicates, that 3 elements of true class $1$ have been predicted as class $0$-instances.   

### Cross Validation
The accuracy on training data should not be applied for model evaluation. Instead a model should be evaluated by determining the accuracy (or other performance figures) on data, which has not been applied for training. Since we have only few labeled data in this example cross-validation is applied for determining the model's accuracy:

from sklearn.model_selection import cross_val_score
clf=GaussianNB()
print(cross_val_score(clf,Income,labels))

### Naive Bayes Classifier for Multidimensional data
In the playground-example above the input-features where only one dimensional: The only input feature has been the annual income of a customer. The 1-dimensional case is quite unusual in practice. In the code-cell below a **Naive Bayes Classifier** is evaluated for multidimensional data. This is just to demonstrate that the same process as applied above for the simple dataset, can also be applied for arbitrary complex datasets.

Again we start from the Bayes Theorem:

$$
P(C_i|\mathbf{x})=\frac{p(\mathbf{x}|C_i)P(C_i)}{P(\mathbf{x})}.
$$

However, the crucial difference to the simple example above is, that not only one random variable $X$ constitutes the input, but many many random variables $X_1,X_2,\ldots,X_N$. I.e a concrete input is a vector 

$$
\mathbf{x}=(x_{i_1},x_{i_1},\ldots,x_{i_N})
$$

The problem is then: **Of what type is the N-dimensional likelihood $p(\mathbf{x}|C_i)$ and how to estimate this likelihood?**

For the general case, where some of the input variables are discrete and others are numeric, there does not exist a joint-likelihood. Therefore, one **naively** assumes that all input variables $X_i$ are independent of each other. Then the N-dimensional likelihood $p(\mathbf{x}|C_i)$ can be factorised into $N$ 1-dimensional likelihoods and these 1-dimenensional likelihoods can be easily estimated from the given training data (as shown above). This is the widely applied **Naive Bayes Classifier:**

$$
P(C_i|\mathbf{x})=\frac{p(\mathbf{x}|C_i)}{P(\mathbf{x})}P(C_i) = \frac{\prod_{j=1}^N p(x_j|C_i)}{P(\mathbf{x})} P(C_i)
$$

 
Below, we apply the [wine dataset](wine.data). This dataset is described [here](wine.names.txt). Actually it is also relatively small, but it contains multidimensional data. 


In the dataset the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of $N=13$ constituents found in each of the three types of wines. The task is to predict the wine-type (first column of the dataset) from the 13 features, that have been obtained in the chemical analysis.

import pandas as pd
wineDataFrame=pd.read_csv("wine.data",header=None)
wineDataFrame

wineData=wineDataFrame.values

print(wineData.shape)

features=wineData[:,1:] #features are in columns 1 to end
labels=wineData[:,0] #class label is in column 0

clf=GaussianNB()
acc=cross_val_score(clf,features,labels,cv=5)

print("Mean Accuracy is ",acc.mean())

