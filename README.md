# Machine-Learning-Building-Linear-and-polynomial-classifiers
Building manually two supervised models and train them for classification purposes.

starting with the linear decision boundary, we first obtain the two features. Then, we add a bias feature for x0 (the first column of the matrix of features is all ones) to express the following equation, which is the features X parameterized by θ:
b*θ0+x1*θ1+x2*θ2
Then applying this equation to the gradient decent, first and before we need to construct the equation of tuning θ, as in the following equation:
(dJ(θ))/dθ=1/m  X^T (sgn(X.θ)-y)
Then multiplying the equation by the learning rate or so-called alpha:
(alpha*dJ(θ))/dθ=1/m  X^T (sgn(X.θ)-y)
After that updating θi in each iteration of gradient decent as in the following equation:
θ≔ θ-(alpha*dJ(θ))/dθ  
And iterating through the G. D. algorithm till we find the theta, which will be a list of parameters θ0, θ1 and θ2.
Results:
After running the first code, to find the decision boundary as a linear function we got the following parameters for θ0, θ1 and θ2 respectively:
[-0.05181422  -1.16222286  0.59785802]
By applying the equation of the sigmoid function to classify the binary class, we can observe that the following equation also holds:
h(theta[0] + theta[1] * x1 + theta[2] * x2) >= 0.5

 then for class C1 or 1 after normalize to 1:
theta[0] + theta[1] * x1 + theta[2] * x2 >= 0

Then plotting the linear decision boundary equation:
x2 = -(theta[0] + theta[1] * x1) / theta[2]

In this part, we repeat the same process as in the previous part, but we modify the decision boundary from linear to quadratic. To do this, we start by constructing the first bias feature and the other two features, x1 and x2, but now squared using the following code snippet:
 
This code will make the X being squared then parameterizing features to produce the following equation:
h(x)= sigmoid(theta[0]* 1  theta[1]* x1^2+ theta[2]* x2^2 )
Then using the previous equation to plot the decision boundary of the following:

Z = theta[0] * 1 + theta[1] * x1**2 + theta[2] * x2**2
