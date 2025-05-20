import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, bobot, bias):

    self.bobot = bobot
    self.bias = bias

  def feedForward(self, inputan):
    # x1, x2 = inputan
    # w1, w2 = self.bobot
    # output = x1 * w1 + x2 * w2 + self.bias
    x = inputan
    w = self.bobot

    output = np.dot(w, x) + self.bias

    y_pred = sigmoid(output)
    return y_pred

bobot = np.array([2, 3])
bias = 1
neuron = Neuron(bobot, bias)
data_input = np.array([1, 1])
hasil = neuron.feedForward(data_input)
print(hasil)

# Neural Network
class MyNetwork():
  def __init__(self, ):
    self.w1 = 2
    self.w2 = 3
    self.w3 = 2
    self.w4 = 3
    self.w5 = 1
    self.w6 = 2
    self.b1 = 1
    self.b2 = 1
    self.b3 = 1

  def feedForward(self, inputan): 
    x1, x2 = inputan
    
    h1 = sigmoid(x1 + self.w1 + x2 * self.w2 + self.b1)
    h2 = sigmoid(x1 + self.w3 + x2 * self.w4 + self.b2)

    o1 = sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)

    return o1
  
# Min-max
def scalling(x):
  return (x - np.min(x)) / (np.max(x) - np.min(x))

# data
data = np.array([
  [170, 80, 1],
  [172, 181, 1],
  [160, 64, 0],
  [155, 59, 0],
  [167, 74, 0],
  [156, 57, 0]
], dtype = float)

print('Before:\n', data)  
data[:,0] = scalling(data[:,0])
data[:,1] = scalling(data[:,1])
# print('After:\n', data)

# Fungsi MSE
def mse_loss(y_true, y_pred):
  return ((y_true - y_pred)**2).mean()



jst = MyNetwork()

epochs = 1000
