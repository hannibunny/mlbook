# Support Vector Machines (SVM)

Support Vector Machines (SVM) have been introduced in {cite}`cortes95`. They can be applied for supervised machine learning, both for classification and regression tasks. Depending on the selected hyperparameter `kernel` (linear, polynomial or rbf), SVMs can learn linear or non-linear models. Other positive features of SVMs are

* many different types of functions can be learned by SVMs, i.e. they have a low bias (for non-linear kernels)
* they scale good with high-dimensional data
* only a small set of hyperparameters must be configured
* overfitting/generalisation can easily be controlled, regularisation is inherent


## SVMs for Classification

SVMs learn class boundaries, which discriminate a pair of classes. In the case of non-binary classification with $K$ classes, $K$ class boundaries are learned. Each of which discriminates one class from all others. In any case, the learned class-boundaries are linear hyperplanes. 

* In the case of a **linear kernel**, each learned (d-1)-hyperplane discriminates the d-dimensional space into two subspaces. $d$ is the dimesionality of the original space, i.e. the number of components in the input-vector $x$.
* In the case of a **non-linear kernel**, the original d-dimensional input data is virtually transformed into a higher-dimensional space with $m > d$ dimensions. In this m-dimensional space training data is hopefully linear-separable. The learnd (m-1)-dimensional hyperplane linearly discriminates the m-dimensional space. But the back-transformation of this hyperplane is a non-linear discrimator in the original d-dimensional space.  

The picture below sketches data, which is not linear-separable in the original 2-dimensional space. However, there exists a transformation into a 3-dimensional space, in which the given training data can be discriminated by a 2-dimensional hyperplane.   

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/svmtransform.jpg" style="width:600px" align="center">
<figcaption>
<em>Data, which is not linear-separable in the original 2-dimensional space may be linear separable in a higher dimensional space.</em>
</figcaption>
</figure>

Another important property of SVM classifiers is that **they found good class-boundaries**. In order to understand what is meant by *good class-boundary* take a look at the picture below. The 4 subplots contain the same training-data but four different class-boundaries, each of which discriminates the given training data error-free. The question is *which of these discriminantes is the best?* The discriminantes in the right column are not robust, because in some regions the datapoints are quite close to the boundary. The discriminant in the top left subplot is the most robust, i.e. the one which generalizes best, because the training-data-free range around the discriminant is maximal. **A SVM classifier actually finds such robust class-boundaries by maximizing the training-data-free range around the discriminant.** 


<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bestdisk.jpg" style="width:600px" align="center">
<figcaption><em>
The discriminant in the top left subplot is the most robust one, because it has a maximal training-data-free range around it.
</em></figcaption>
</figure>


### Finding Robust Discriminants

As mentioned above, SVM classifiers find robust discriminants in the sense that the discriminant is determined such that it not only separates the given training-data well, but also has a maximal training-data free range around it. In this subsection it is shown, how such discriminantes are learned from data. To illustrate this we apply the example depicted below. We have a set of 8 training-instances, partitioned into 2 classes. The task is to find a robust discriminant with the properties mentioned above.

<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/svmExampleIntro.png" style="width:600px" align="center">
<figcaption><em>
Example: Learn a good discriminant from the set of 8 training-instances 
</em></figcaption>
</figure>


As ususal in supervised machine learning, we start from a set of $N$ labeled training instances

$$ 
T=\{\mathbf{x}_p,r_p\}_{p=1}^N,
$$

where $\mathbf{x}_p$ is the p.th input-vector and $r_p$ is the corresponding label. In the context of binary SVMs the label-values are:

* $r_p=-1$, if $\mathbf{x}_p \in C_1$ 
* $r_p=+1$, if $\mathbf{x}_p \in C_2$
	
The **training goal** is: Determine the weights $\mathbf{w}=(w_1,\ldots,w_d)$ and $w_0$, such that

\begin{eqnarray}
\mathbf{w}^T \mathbf{x_p} + w_0 \geq +1 & \mbox{ for } & r_p=+1 \nonumber \\
\mathbf{w}^T \mathbf{x_p} + w_0 \leq -1 & \mbox{ for } & r_p=-1 \nonumber 
\label{eq:k2svmklass}
\end{eqnarray}
	
This goal can equivalently be formulated by imposing the following condition
	
$$
r_p (\mathbf{w}^T \mathbf{x_p} + w_0 ) \geq 1
$$

	
to be fullfilled for all training-instances. Note, that this condition defines a **boundary area**, rather than just a **boundary line**, as in other algorithms, where the condition, that must be fullfilled is:

$$
r_p (\mathbf{w}^T \mathbf{x_p} + w_0 ) \geq 0
$$

	
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/svmcomb.jpg" style="width:600px" align="center">
<figcaption><em>
Example Autostichting
</em></figcaption>
</figure>
