import numpy as np

# Definisi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# inisialisasi input layer (input layer => 2 node)
X = np.random.rand(2, 1) # 2 input neurons, 1 sample(column vector)

# inisialisasi bobot dan bias secara random
np.random.seed(42)

# inisialisasi hidden layer (hidden layer => 2 node)
W_hidden = np.random.rand(2, 2) # bobot untuk hidden layer (2x2)
b_hidden = np.random.rand(2, 1) # bias untuk hidden layer (2x1)

# inisialisasi output layer (output layer => 1 node)
W_output = np.random.rand(1, 2) # bobot untuk hidden layer (1x2)
b_output = np.random.rand(1, 1) # bias untuk output layer (1x1)

# --- FeedForward ---
# 1. perhitungan di hiden layer
z_hidden = np.dot(W_hidden, X) + b_hidden # perhitungan formula: z=W.x+b; 
a_hidden = sigmoid(z_hidden)

# 2. perhitungan di output layer
z_output = np.dot(W_output, a_hidden) + b_output # perhitungan formula: z=W.x+b; 
a_output = sigmoid(z_output)

# mencetak hasil 
print("input: \n", X)
print("Hidden Layer Output (a_hidden): \n", a_hidden)
print("Final Output (a_hidden): \n", a_output)
