import numpy as np
import pandas as pd

#sign function
def sign(x):
    return np.where(x >= 0, 1.0, -1.0)  
#softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
pf = pd.read_csv(
    r'C:\Users\kotip\OneDrive\Desktop\MINI PROJECTS 2\IIT M internship\student-lifestyle-and-stress-dataset.csv'
)
pf['Student_Type'] = pf['Student_Type'].map({
    'school': 0,
    'college': 1,
    'working_student':2
})
#fill missing values with mean
pf = pf.fillna(pf.mean())
#features
x=pf[['Student_Type','Sleep_Hours','Study_Hours','Social_Media_Hours','Attendance','Exam_Pressure','Family_Support','Month']].to_numpy()
x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
#target variable
y=pf['Stress_Level'].to_numpy()
batch_size = 1000
number_layers = 3
input_size = x.shape[1]
output_size = len(np.unique(y))
layers=[input_size, 16, 8, output_size]
#initialization of parameters
w,b,gamma,beta,running_mean,running_var = [],[],[],[],[],[]
w = [
    np.random.randn(layers[i], layers[i+1])
    for i in range(number_layers)
]
b = [
    np.random.randn(layers[i+1], 1)
    for i in range(number_layers)
]
gamma = [
    np.ones((1, layers[i+1]))
    for i in range(number_layers - 1)
]

beta = [
    np.zeros((1, layers[i+1]))
    for i in range(number_layers - 1)
]
running_mean = [
    np.zeros((1, layers[i+1]))
    for i in range(number_layers - 1)
]
running_var = [
    np.ones((1, layers[i+1]))
    for i in range(number_layers - 1)
]
#batch normalization function
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5,training=True,momentum=0.9):
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    if training:
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        out = gamma * x_norm + beta
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    return out, batch_var, x_norm
#cross entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss
#shuffle function
def shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]
#epochs
epochs = 500
mem=0.9
lr=0.05
#training loop
split = int(0.8 * len(x))
x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]
v_w = [np.zeros_like(wi) for wi in w]
v_b = [np.zeros_like(bi) for bi in b]
for i in range(epochs):
    x_train, y_train = shuffle(x_train, y_train)
    for j in range(0, len(x_train), batch_size):
        Lookahead_w=[w-mem*dw for w, dw in zip(w, [np.zeros_like(wi) for wi in w])]
        Lookahead_b=[b-mem*db for b, db in zip(b, [np.zeros_like(bi) for bi in b])]
        lookahead_gamma=[gamma-mem*dgamma for gamma, dgamma in zip(gamma, [np.zeros_like(gi) for gi in gamma])]
        lookahead_beta=[beta-mem*dbeta for beta, dbeta in zip(beta, [np.zeros_like(bi) for bi in beta])]
        x_batch = x_train[j:j+batch_size]
        y_batch = y_train[j:j+batch_size]
        #forward pass
        activations = [x_batch]
        a = x_batch
        for k in range(number_layers):
            w_bin=sign(w[k])
            z = np.dot(a, w_bin) + b[k].T
            if k < number_layers - 1:
                z, _, _ = batch_norm(z, gamma[k], beta[k], running_mean[k], running_var[k], training=True)
                a = sign(z)
            else:
                a = softmax(z)
            activations.append(a)
        #backward pass
        loss = cross_entropy_loss(y_batch, a)
        for k in range(number_layers-1, -1, -1):
            if k == number_layers - 1:
                dz = a
                dz[range(len(y_batch)), y_batch] -= 1
            else:
                dz = np.dot(dz, w[k+1].T)
                dz = dz * (np.abs(activations[k+1]) <= 1)
                dz, batch_var, x_norm = batch_norm(dz, gamma[k], beta[k], running_mean[k], running_var[k], training=False)
                dgamma = np.sum(dz * x_norm, axis=0)
                dbeta = np.sum(dz, axis=0)
                v_dgamma = mem * (gamma[k] - lookahead_gamma[k]) + dgamma
                v_dbeta = mem * (beta[k] - lookahead_beta[k]) + dbeta
                gamma[k] -= lr * v_dgamma
                beta[k] -= lr * v_dbeta
            dw = np.dot(activations[k].T, dz) / batch_size
            db = np.sum(dz, axis=0) / batch_size
            v_w[k] = mem * v_w[k] + dw
            v_b[k] = mem * v_b[k] + db.reshape(-1,1)

            w[k] -= lr * v_w[k]
            b[k] -= lr * v_b[k]
#testing loop
a = x_test
for k in range(number_layers):
    w[k] = sign(w[k])
    z = np.dot(a, w[k]) + b[k].T
    if k < number_layers - 1:
        z, _, _ = batch_norm(z, gamma[k], beta[k], running_mean[k], running_var[k], training=False)
        a = sign(z)
    else:
        a = softmax(z)
y_pred = np.argmax(a, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
    
        