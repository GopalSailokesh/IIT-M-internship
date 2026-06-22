import numpy as np
import matplotlib.pyplot as plt

def sigmoid(w, b, x):
    return 1 / (1 + np.exp(-(w * x + b)))

x = [0.5, 2.5]
y = [0.2, 0.9]

def error(w, b):
    err = 0.0
    for i, j in zip(x, y):
        err += 0.5 * (sigmoid(w, b, i) - j) ** 2
    return err

def grad_w(w, b):
    grad = 0.0
    for i, j in zip(x, y):
        fx = sigmoid(w, b, i)
        grad += (fx - j) * fx * (1 - fx) * i
    return grad

def grad_b(w, b):
    grad = 0.0
    for i, j in zip(x, y):
        fx = sigmoid(w, b, i)
        grad += (fx - j) * fx * (1 - fx)
    return grad

def do_gradient_descent():
    w = -2
    b = -2
    eta = 1.0
    epochs = 1000

    for epoch in range(epochs):
        dw = grad_w(w, b)
        db = grad_b(w, b)

        w = w - eta * dw
        b = b - eta * db

    return w, b

w, b = do_gradient_descent()

W = np.linspace(-6, 6, 100)
B = np.linspace(-6, 6, 100)
W, B = np.meshgrid(W, B)

L = np.zeros_like(W)

for i in range(len(x)):
    L += 0.5 * (sigmoid(W, B, x[i]) - y[i]) ** 2

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, B, L, cmap='coolwarm', alpha=0.8)

ax.scatter(
    w,
    b,
    error(w, b),
    color='red',
    s=100,
    label='Gradient Descent Point'
)

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Error')
ax.set_title('Gradient Descent on Error Surface')
ax.legend()

plt.colorbar(surf)
plt.show()

print("w =", w)
print("b =", b)
print("final error =", error(w, b))

