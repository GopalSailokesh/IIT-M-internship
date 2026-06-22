"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2])
y = np.array([2, 4])

w_range = np.linspace(-6, 6, 100)
b_range = np.linspace(-6, 6, 100)

W, B = np.meshgrid(w_range, b_range)

L = np.zeros_like(W)

for i in range(len(x)):
    L += (W * x[i] + B - y[i]) ** 2

L = L / len(x)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    W,
    B,
    L,
    cmap='coolwarm',
    alpha=0.8
)

best_loss = float('inf')
best_w = 0
best_b = 0

for _ in range(20):
    w = np.random.uniform(-6, 6)
    b = np.random.uniform(-6, 6)

    loss = np.mean((w * x + b - y) ** 2)

    ax.scatter(
        w,
        b,
        loss,
        color='red',
        s=30
    )

    if loss < best_loss:
        best_loss = loss
        best_w = w
        best_b = b

ax.scatter(
    best_w,
    best_b,
    best_loss,
    color='green',
    s=100,
    label='Best Point'
)

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('error')
ax.set_title('Random Search on Error Surface')

ax.legend()

plt.colorbar(surf)
plt.show()

print("Best w =", best_w)
print("Best b =", best_b)
print("Minimum loss =", best_loss)
"""
