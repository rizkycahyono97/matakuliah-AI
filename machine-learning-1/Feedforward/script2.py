import numpy as np

# 
class Neuron:
    def __init__(self, bobot, bias):
        self.bobot = bobot
        self.bias = bias

    def feedForward(self, inputan):
        x1, x2 = inputan
        w1, w2 = self.bobot
        output = x1 * w1 + x2 * w2 + self.bias
        
        return output