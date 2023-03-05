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
        return 1
    elif label == "C2":
        return 0
    # add more cases if necessary
    else:
        return None


Y = list(map(encode_label, Y))

X = np.c_[np.ones((len(X), 1)), X]
# initialize theta
theta = np.zeros(X.shape[1])
# Fit logistic regression model
model = LogisticRegression(C=1.0, solver="lbfgs")
model.fit(X, Y)
theta = model.coef_[0]
x1 = train["x1"].values
x2 = train["x2"].values
# h(theta[0] + theta[1] * x1 + theta[2] * x2) >= 0.5
# theta[0] + theta[1] * x1 + theta[2] * x2 = 0
#  x2 = -(theta[0] + theta[1] * x1) / theta[2]


line_x1 = np.linspace(-1, 1, 100)
x2 = -(theta[1] * line_x1 + theta[0]) / theta[2]
print(theta)


def compute_error(x1, x2, y):
    val = theta[0] + theta[1] * x1 + theta[2] * x2
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
x1_train = list(X[:, 1])
x2_train = list(X[:, 2])
y_train = list(Y)
count = 0
for i in range(len(x1_train)):
    count += compute_error(x1_train[i], x2_train[i], y_train[i])
accuracy = count / len(y_train)
print("Train Accuracy: ", accuracy)


# plot the points and the decision boundary
plt.scatter(X[:, 1], X[:, 2], c=Y)
# for test data
# plt.scatter(X_t[:, 0], X_t[:, 1], c=Y_t)
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.plot(line_x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
