# IMPORT LIBRARIES
import numpy as np
import idx2numpy as idx2np
import matplotlib.pyplot as plt

# LOAD TRAIN DATA
data = idx2np.convert_from_file(
    r"C:\Users\kotip\OneDrive\Desktop\Projects\MINI PROJECTS 2\IIT M internship\train-images.idx3-ubyte"
)
labels = idx2np.convert_from_file(
    r"C:\Users\kotip\OneDrive\Desktop\Projects\MINI PROJECTS 2\IIT M internship\train-labels.idx1-ubyte"
)
x = data.reshape(data.shape[0], -1).T / 255.0
y = labels

# ACTIVATION FUNCTIONS
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def binarize(x):
    return np.where(x >= 0, 1.0, -1.0)

def ste_gradient(x):
    return (np.abs(x) <= 1).astype(np.float32)

# BATCH NORM
def batch_norm(x, gamma, beta, moving_mean, moving_var,
               eps=1e-5, training=True, momentum=0.8):   
    if training:
        mean = np.mean(x, axis=1, keepdims=True)
        var  = np.var(x,  axis=1, keepdims=True)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var  = momentum * moving_var  + (1 - momentum) * var
    else:
        mean = moving_mean
        var  = moving_var

    x_hat = (x - mean) / np.sqrt(var + eps)

    if gamma is not None and beta is not None:
        out = gamma * x_hat + beta
    else:
        out = x_hat

    cache = (x_hat, gamma, beta, mean, var, eps)

    return out, cache, moving_mean, moving_var


def one_hot(y, num_classes=10):
    y_onehot = np.zeros((num_classes, len(y)))
    y_onehot[y, np.arange(len(y))] = 1
    return y_onehot


# NETWORK PARAMETERS

input_size  = 784
output_size = 10

layer_sizes = [784, 4096,4096, 4096, 10]  
number_layers = len(layer_sizes) - 1
lr         = 0.09  
epochs     = 10
batch_size = 128
mem        = 0.8     

#  INITIALIZATION
W           = []
b           = []
gamma_list  = []
beta_list   = []
moving_means = []
moving_vars  = []

np.random.seed(42)

for i in range(number_layers):
    W.append(
        np.random.randn(layer_sizes[i], layer_sizes[i + 1])
        * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1])) 
    )
    b.append(np.zeros((layer_sizes[i + 1], 1)))

for i in range(number_layers - 1):
    gamma_list.append(np.ones((layer_sizes[i + 1], 1)))
    beta_list.append(np.zeros((layer_sizes[i + 1], 1)))
    moving_means.append(np.zeros((layer_sizes[i + 1], 1)))
    moving_vars.append(np.ones((layer_sizes[i + 1], 1)))

Vw     = [np.zeros_like(w) for w in W]
Vb     = [np.zeros_like(bi) for bi in b]
Vgamma = [np.zeros_like(g) for g in gamma_list]   
Vbeta  = [np.zeros_like(bt) for bt in beta_list]  

# FORWARD PROPAGATION

def feedforward(X, W_arg, b_arg, gamma_arg, beta_arg, training=True):
    h          = [X]
    z_values   = []
    caches     = []   
    bn_outputs = []   
    for i in range(number_layers):
        W_bin = binarize(W_arg[i])
        z = W_bin.T @ h[-1] + b_arg[i]
        z_values.append(z)

        if i == number_layers - 1:
            h.append(softmax(z))
        else:
            z_bn, cache, moving_means[i], moving_vars[i] = batch_norm(
                z, gamma_arg[i], beta_arg[i],
                moving_means[i], moving_vars[i],
                training=training
            )
            caches.append(cache)        
            bn_outputs.append(z_bn)     
            h.append(binarize(z_bn))

    return h, z_values, caches, bn_outputs
# BACKPROPAGATION
def backpropagation(h, y_true, z_values, caches, bn_outputs,
                    W_arg, gamma_arg, beta_arg, eps=1e-5):
    global Vw, Vb, Vgamma, Vbeta

    m  = y_true.shape[1]
    dz = h[-1] - y_true        
    grad_gamma_list = []
    grad_beta_list  = []

    for k in reversed(range(number_layers)):
        dw = (h[k] @ dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        Vw[k] = mem * Vw[k] + lr * dw
        Vb[k] = mem * Vb[k] + lr * db
        W[k] -= Vw[k]
        b[k] -= Vb[k]
        W[k]  = np.clip(W[k], -1, 1)

        if k > 0:
            W_bin_next = binarize(W_arg[k])
            G = W_bin_next @ dz                             
            ste_mask = (np.abs(bn_outputs[k - 1]) <= 1).astype(float)
            G *= ste_mask                    
            cache  = caches[k - 1]
            a_hat  = cache[0]   
            var    = cache[4]   
            g_i    = gamma_arg[k - 1]
            dg = np.sum(G * a_hat, axis=1, keepdims=True) / m
            dbt = np.sum(G, axis=1, keepdims=True) / m
            grad_gamma_list.append(dg)
            grad_beta_list.append(dbt)

            Vgamma[k - 1] = mem * Vgamma[k - 1] + lr * dg
            Vbeta[k - 1]  = mem * Vbeta[k - 1]  + lr * dbt
            gamma_list[k - 1] -= Vgamma[k - 1]
            beta_list[k - 1]  -= Vbeta[k - 1]
            Q  = G * g_i
            dz = (Q
                  - np.mean(Q, axis=1, keepdims=True)
                  - a_hat * np.mean(Q * a_hat, axis=1, keepdims=True)
                 ) / np.sqrt(var + eps)

# PREDICTION & TEST 
def predict(X):
    h, _, _, _ = feedforward(X, W, b, gamma_list, beta_list, training=False)
    return np.argmax(h[-1], axis=0)

def test(X_test, y_test):
    return np.mean(predict(X_test) == y_test)
loss_history      = []
train_acc_history = []
test_acc_history  = []

#  TRAINING LOOP
def train(X, y, X_test, y_test):
    samples = X.shape[1]

    for epoch in range(epochs):
        indices    = np.random.permutation(samples)
        X_shuffled = X[:, indices]
        y_shuffled = y[indices]
        total_loss = 0

        for start in range(0, samples, batch_size):
            end = start + batch_size
            lookahead_W     = [w  - mem * v  for w,  v in zip(W,          Vw)]
            lookahead_b     = [bi - mem * v  for bi, v in zip(b,          Vb)]
            lookahead_gamma = [g  - mem * v  for g,  v in zip(gamma_list, Vgamma)]
            lookahead_beta  = [bt - mem * v  for bt, v in zip(beta_list,  Vbeta)]

            X_batch  = X_shuffled[:, start:end]
            y_batch  = y_shuffled[start:end]
            y_onehot = one_hot(y_batch)

            h, z_values, caches, bn_outputs = feedforward(
                X_batch,
                lookahead_W, lookahead_b,
                lookahead_gamma, lookahead_beta
            )

            loss = -np.sum(y_onehot * np.log(h[-1] + 1e-9)) / y_onehot.shape[1]
            total_loss += loss

            backpropagation(
                h, y_onehot, z_values, caches, bn_outputs,
                lookahead_W, lookahead_gamma, lookahead_beta
            )

        avg_loss  = total_loss / (samples // batch_size)
        test_acc  = test(X_test, y_test)
        train_acc = np.mean(predict(X) == y)

        loss_history.append(avg_loss)
        train_acc_history.append(train_acc * 100)
        test_acc_history.append(test_acc * 100)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss={avg_loss:.4f} | "
            f"Training Accuracy={train_acc*100:.2f}% | "
            f"Testing Accuracy={test_acc*100:.2f}%"
        )


# LOAD TEST DATA & RUN

y_test = idx2np.convert_from_file(
    r"C:\Users\kotip\OneDrive\Desktop\Projects\MINI PROJECTS 2\IIT M internship\t10k-labels.idx1-ubyte"
)
x_test = idx2np.convert_from_file(
    r"C:\Users\kotip\OneDrive\Desktop\Projects\MINI PROJECTS 2\IIT M internship\t10k-images.idx3-ubyte"
)
x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

train(x, y, x_test, y_test)

test_accuracy = test(x_test, y_test)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")

# PLOT
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_acc_history, label="Training Accuracy")
plt.plot(range(1, epochs + 1), test_acc_history,  label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.grid(True)
plt.show()