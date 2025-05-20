import torch.nn as nn

class IrisMLP(nn.Module):
    def __init__(self):
        super(IrisMLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)   # 4 input features → 10 hidden neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)   # 3 output classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
