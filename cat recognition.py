import numpy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train, num_px*num_px*3)

train_set_x = train_set_x_flatten/255

def sigmoid(z):
    s = 1/(1 + numpy.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = numpy.zeros((dim,1))
    b = 0
    return w,b

def propagate(w, b, X, Y):
    m = X.shape[0]
    A = sigmoid(numpy.dot(w.transpose(), X.transpose()) + b)
    cost = numpy.sum(Y*numpy.log(A) + (1-Y)*numpy.log(1-A))
    cost = -cost/m

    dw = (numpy.dot(X.transpose(), (A-Y).transpose()))/m
    db = (numpy.sum(A-Y))/m

    grads = {"dw":dw,
             "db":db}

    return grads,cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100 == 0:
            costs.append(cost)

        if print_cost and i%10 == 0:
            print("Cost after interation %i: %f" %(i,cost))

    params = {"w":w,
              "b":b}

    grads = {"dw":dw,
             "db":db}
    return params, grads, costs

def predict(w, b ,X):
    m = X.shape[0]
    Y_prediction = numpy.zeros((1,m))

    A = sigmoid(numpy.dot(w.transpose(), X.transpose()) + b)

    for i in range(A.shape[1]):
        if A[0][i]<=0.5:
            Y_prediction[0][i] = 0
        if A[0][i]>=0.5:
            Y_prediction[0][i] = 1

    return Y_prediction

def model(X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b = initialize_with_zeros(X_train.shape[1])

    parameters,grads,costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - numpy.mean(numpy.abs(Y_prediction_train - Y_train)) * 100))

    d = {"costs": costs,
         "Y_prediction_train": Y_prediction_train,
         "w":w,
         "b":b,
         "learning_rate":learning_rate,
         "num_iterations":num_iterations}
    return d

d = model(train_set_x, train_set_y, num_iterations=2000, learning_rate=0.005, print_cost=False)

my_image = "cat.jpg"
image = numpy.array(ndimage.imread(my_image, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3))
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(numpy.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(numpy.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

