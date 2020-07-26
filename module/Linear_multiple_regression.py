import numpy as np

def Linear_multiple_regression(inputs, objects):

    w = np.dot(np.dot(np.linalg.inv(np.dot(inputs.T, inputs)), inputs.T), objects)

    return w
