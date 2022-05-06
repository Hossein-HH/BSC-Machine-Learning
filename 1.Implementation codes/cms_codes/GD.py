import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([
    (1, 80),
    (2, 170),
    (3, 100),
    (3, 220),
    (4, 200),
    (4, 270),
    (5, 500)
])

x = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1].reshape(-1, 1)


def add_intercept_ones(X):
    intercept_ones = np.ones((len(X), 1))  # results in array( [ [1],..,[1] ] )
    X_b = np.c_[intercept_ones, X]
    return X_b


X_b = add_intercept_ones(x)


def gradient_descent(x, y, w, epochs, alpha, eps=0.001):
    epoch = 1
    loss_history = []
    while True:
        y_hat = np.dot(x, w)  # Xw ==> X@w
        error = y_hat - y
        loss = (1.0 / x.shape[0]) * error.T @ error  # MSE(loss) => 1/(n)*(y_hat-y).T@(y_hat-y)
        loss_history.append(loss[0][0])
        w -= 2 * alpha * x.T @ error  # w -= alpha*X.T(Xw-y)
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}]: loss = {loss}')
        if epoch > epochs or loss < eps:
            break
        epoch += 1
    return w, loss_history


w = np.random.randn(x.shape[1] + 1, 1)
xx = add_intercept_ones(x)
w_hat, loss = gradient_descent(xx, y, w, 800, 0.01)
print(f'My weights: {w_hat[:, 0]}')
w_star = np.linalg.inv(xx.T @ xx) @ xx.T @ y
print(f'Math weights: {w_star[:, 0]}')

plt.plot(dataset[:, 0], dataset[:, 1], 'g*')
x = np.linspace(0.5, 6, 100).reshape(-1, 1)
x = add_intercept_ones(x)
y = x @ w_hat
yy = x @ w_star
plt.plot(x[:, 1], y[:, 0], 'b', label='My weights')
plt.plot(x[:, 1], yy[:, 0], 'r', label='Math weights')
plt.legend()
plt.figure()
plt.plot(list(range(len(loss))), loss)
plt.show()
