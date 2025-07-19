import random
import time

# Generate synthetic dataset
x = [random.random() for _ in range(100000)]
y = [3 * xi + 4 + random.gauss(0, 0.1) for xi in x]

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 10
n = len(x)

start = time.time()

# Gradient Descent
for epoch in range(epochs):
    dm = 0
    db = 0
    for i in range(n):
        y_pred = m * x[i] + b
        error = y[i] - y_pred
        dm += -2 * x[i] * error
        db += -2 * error
    m -= (dm / n) * learning_rate
    b -= (db / n) * learning_rate
    print(f"Epoch {epoch+1}: m = {m}, b = {b}")

end = time.time()
print(f"\nExecution Time: {(end - start)*1000:.2f} ms")
