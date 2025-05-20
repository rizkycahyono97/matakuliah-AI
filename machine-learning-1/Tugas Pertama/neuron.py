import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, bobot, bias):

    self.bobot = bobot
    self.bias = bias

  def feedForward(self, inputan):
    x1, x2 = inputan
    w1, w2 = self.bobot
    output = x1 * w1 + x2 * w2 + self.bias

    y_pred = sigmoid(output)
    return y_pred

bobot = np.array([2, 3])
bias = 1
neuron = Neuron(bobot, bias)

data_input = np.array([1, 1])

hasil = neuron.feedForward(data_input)

print(hasil)