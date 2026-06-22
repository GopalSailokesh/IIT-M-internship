import numpy as np
import idx2numpy as idx2np
#importing dataset
data=idx2np.convert_from_file(r"C:\Users\kotip\OneDrive\Desktop\MINI PROJECTS 2\IIT M internship\train-images.idx3-ubyte")
labels=idx2np.convert_from_file(r"C:\Users\kotip\OneDrive\Desktop\MINI PROJECTS 2\IIT M internship\train-labels.idx1-ubyte")
x=data.reshape(data.shape[0], -1).T / 255.0
y=labels
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#derivative of sigmoid function
def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
#softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
#binarization function
def binarize(x):
    return np.where(x >= 0, 1, -1)
#batch normalization function
def batchnorm(x, gamma, beta, running_mean, running_var, momentum=0.9, training=True, eps=1e-5):
    if training:
        mu = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)
        # Update running statistics
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    cache = (x_hat, gamma, beta, running_mean, running_var, eps)
    return gamma * x_hat + beta, cache  

# initialization of Layers
input_size = x.shape[0]
output_size = len(np.unique(y))
hidden_size = 32
layer_sizes = [input_size, hidden_size, hidden_size, hidden_size, output_size]
Batch_size = 32
number_layers = len(layer_sizes) - 1
lr = 0.01
epochs = 10

#initialization of W, b, gamma, beta, running_mean, running_var
W, b, gamma, beta, running_means, running_vars = [], [], [], [],[], []
np.random.seed(42)
W, b, gamma, beta, running_means, running_vars = [], [], [], [], [], []

np.random.seed(42)

for i in range(number_layers):
    W.append(
        np.random.randn(layer_sizes[i], layer_sizes[i+1]) *
        np.sqrt(1 / layer_sizes[i])
    )
    b.append(np.zeros((layer_sizes[i+1], 1)))
for i in range(number_layers - 1):
    gamma.append(np.ones((layer_sizes[i+1], 1)))
    beta.append(np.zeros((layer_sizes[i+1], 1)))
    running_means.append(np.zeros((layer_sizes[i+1], 1)))
    running_vars.append(np.ones((layer_sizes[i+1], 1)))

#BinaryNormalization function
def binary_normalization(x, gamma, beta, running_mean, running_var, momentum=0.9, training=True, eps=1e-5):
    batch_mean = np.mean(x, axis=1, keepdims=True)
    batch_var = np.var(x, axis=1, keepdims=True)
    if training:
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        out = gamma * x_norm + beta
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    return binarize(out), batch_mean, batch_var, x_norm

#cross entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

#farward pass function
def feedforward(W, x, b):
    h = [x]
    a = []
    bn_cache = []
    for i in range(number_layers):
        Wb = binarize(W[i])
        a.append(Wb.T @ h[-1] + b[i])
        if i == number_layers - 1:
            h.append(softmax(a[-1]))
        else:
            z_norm, batch_mean, batch_var, x_norm = binary_normalization(a[-1], gamma[i], beta[i], running_means[i], running_vars[i], training=True)
            bn_cache.append((batch_mean, batch_var, x_norm))
            h.append(sigmoid(z_norm))
    return h, a, bn_cache

#backpropagation function
def backpropagation(W, h, y, a, bn_cache):  
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    for k in range(number_layers - 1, -1, -1):
        if k < number_layers - 1:
            dz = np.dot(grad_a[0], W[k+1].T)
            dz = dz * h[k+1] * (1 - h[k+1])  # Derivative of binarized activation
            batch_mean, batch_var, x_norm = bn_cache[k]
            dgamma = np.sum(dz * x_norm, axis=0, keepdims=True)
            dbeta = np.sum(dz, axis=0, keepdims=True)
            grad_a.insert(0, dz)
            gamma[k] -= lr * dgamma
            beta[k] -= lr * dbeta   
        else:
            dz = grad_a[0]
            grad_a.insert(0, dz)
        dw = np.dot(h[k], grad_a[0].T) / batch_size
        db = np.sum(grad_a[0], axis=0, keepdims=True)
        W[k] -= lr * dw
        b[k] -= lr * db.T   
def one_hot(y, num_classes=10):
    y_onehot = np.zeros((num_classes, len(y)))
    y_onehot[y, np.arange(len(y))] = 1
    return y_onehot
#training loop
def train(X, y, W, b, gamma, beta, running_means, running_vars, epochs=10, batch_size=32):
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)
        X = X[:, indices]
        y = y[indices]
        
        for j in range(0, X.shape[1], batch_size):
            x_batch = X[:, j:j+batch_size]
            y_batch = y[j:j+batch_size]
            y_batch_onehot = one_hot(y_batch)
            h, a, bn_cache = feedforward(W, x_batch, b)
            backpropagation(W, h, y_batch_onehot, a, bn_cache)
#testing loop
def test(X, y, W, b):
    h, _, _ = feedforward(W, X, b)
    predictions = np.argmax(h[-1], axis=0)
    accuracy = np.mean(predictions == y)
    return accuracy 
#shuffle function
def shuffle(x, y):  
    indices = np.arange(x.shape[1])
    np.random.shuffle(indices)
    x = x[:, indices]
    y = y[indices]
    return x, y 
print("Training started...")
train(x, y, W, b, gamma, beta, running_means, running_vars, epochs=epochs, batch_size=Batch_size)
print("Training completed.")
accuracy = test(x, y, W, b)
print(f"Training Accuracy: {accuracy * 100:.2f}%")