import math
def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s
#print(basic_sigmoid(3))

import numpy as np
x = np.array([1, 2, 3])
#print(np.exp(x))
#print(x + 3)

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

#print(sigmoid(3))
#print(sigmoid(x))

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function whth respect to its input x.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

#print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v
# 3*3*2 matrix
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

# v = image2vector(image)
# print(v.shape[0], v.shape[1])
# print ("image2vector(image) = " + str(image2vector(image)))
#print(image.shape[2])

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. 
    """
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x / x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
#print('normalizeRows(x) = ' + str(normalizeRows(x)))

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Code works for both a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n, m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
#print("softmax(x) = " + str(softmax(x)))


import time 

# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

# ### CLASSIC DOT PRODUCT
# tic = time.process_time()
# dot = 0
# for i in range(len(x1)):
#     dot += x1[i] * x2[i]
# toc = time.process_time()
# print("dot = " + str(dot) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + 'ms')

# ### CLASSIC OUTER PRODUCT
# tic = time.process_time()
# outer = np.zeros((len(x1), len(x2)))
# for i in range(len(x1)):
#     for j in range(len(x2)):
#         outer[i, j] = x1[i] * x2[j]
# toc = time.process_time()
# print("outer = " + str(outer) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + "ms")

# ### CLASSIC ELEMENTWISE
# tic = time.process_time()
# mul = np.zeros(len(x1))
# for i in range(len(x1)):
#     mul[i] = x1[i] * x2[i]
# toc = time.process_time()
# print("elementwise multiplication = " + str(mul) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + "ms")

# ### CLASSIC GENERAL DOT PRODUCT
# W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
# tic = time.process_time()
# gdot = np.zeros(W.shape[0])
# for i in range(W.shape[0]):
#     for j in range(len(x1)):
#         gdot[i] += W[i,j]*x1[j]
# toc = time.process_time()
# print("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

# ### VECTORIZED DOT PRODUCT OF VECTORS
# tic = time.process_time()
# dot = np.dot(x1, x2)
# toc = time.process_time()
# print("dot = " + str(dot) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + 'ms')

# ### VECTORIZED OUTER PRODUCT
# tic = time.process_time()
# outer = np.outer(x1, x2)
# toc = time.process_time()
# print("outer = " + str(outer) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + "ms")

# ### VECTORIZED ELEMENTWISE MULTIPLICATION
# tic = time.process_time()
# mul = np.multiply(x1, x2)
# toc = time.process_time()
# print("elementwise multiplication = " + str(mul) + "\n ----- Compution time = " + str(1000 * (toc - tic)) + "ms")

# ### VECTORIZED GENERAL DOT PRODUCT
# W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
# tic = time.process_time()
# gdot = np.dot(W, x1)
# toc = time.process_time()
# print("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- Vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.sum(np.dot(yhat - y, yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))