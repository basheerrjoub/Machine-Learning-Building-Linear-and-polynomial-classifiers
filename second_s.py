import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Train Dataset
train = pd.read_csv("train.csv")
X = train[["x1", "x2"]].values
Y = train["class"].values

# C1 and C2 --> 1, 0
def encode_label(label):
    if label == "C1":
        return 0
    elif label == "C2":
        return 1


Y = np.array(list(map(encode_label, Y)))
xtemp = X

# add squared features after one
X = np.c_[np.ones((len(X), 1)), np.square(X)]


# fitting the  logistic regression model
model = LogisticRegression(
    solver="liblinear",
    multi_class="auto",
    max_iter=10000,
    C=1000,
    penalty="l2",
    fit_intercept=False,
)
model.fit(X, Y)
theta = model.coef_[0]
print(theta)

# h(x) = sigmoid(theta[0] * 1 + theta[1] * x1 + theta[2] * x2 + theta[3] * x1^2 + theta[4] * x2^2)
# writing then x2 in terms of others we get the following
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
x1, x2 = np.meshgrid(x1, x2)
Z = theta[0] * 1 + theta[1] * x1**2 + theta[2] * x2**2


def compute_error(x1, x2, y):
    val = theta[0] * 1 + theta[1] * x1**2 + theta[2] * x2**2
    predicted = 0
    if val >= 0:
        predicted = 1
    if predicted == y:
        return 1
    else:
        return 0


# Test Dataset
test = pd.read_csv("test.csv")
X_t = test[["x1", "x2"]].values
Y_t = test["class"].values
Y_t = list(map(encode_label, Y_t))

# Computing the Accuracy of the results:
x1_t = list(X_t[:, 0])
x2_t = list(X_t[:, 1])
y_t = list(Y_t)
count = 0
for i in range(len(x1_t)):
    count += compute_error(x1_t[i], x2_t[i], y_t[i])
accuracy = count / len(y_t)
print("Test Accuracy: ", accuracy)


# Computing the Accuracy of Trainning:
x1_train = list(xtemp[:, 0])
x2_train = list(xtemp[:, 1])
y_train = list(Y)
count = 0
for i in range(len(x1_train)):
    count += compute_error(x1_train[i], x2_train[i], y_train[i])
accuracy = count / len(y_train)
print("Train Accuracy: ", accuracy)


# plot the points and the decision boundary
plt.scatter(xtemp[:, 0], xtemp[:, 1], c=Y)
# Plot tested data scatter
# plt.scatter(X_t[:, 0], X_t[:, 1], c=Y_t)
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.contour(x1, x2, Z, [0])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
