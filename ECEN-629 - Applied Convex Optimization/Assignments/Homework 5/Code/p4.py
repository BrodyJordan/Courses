import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.svm import SVC

# --- Part (a) Data Generation ---
import numpy as np
import pandas as pd

# 1. Setup Parameters from Problem 4(a)
np.random.seed(42)
n_samples = 800
pi1, pi2 = 0.4, 0.6
mu1 = [0, 0]
cov1 = [[1, 0], [0, 0.5]] # Variances: 1, 0.5
mu2 = [2, 1]
cov2 = [[np.sqrt(2), 0], [0, 2]] # Variances: sqrt(2), 2

# 2. Generate Data
labels = np.random.choice([-1, 1], size=n_samples, p=[pi1, pi2])
data = []

for l in labels:
    if l == -1:
        pt = np.random.multivariate_normal(mu1, cov1)
        data.append([pt[0], pt[1], -1])
    else:
        pt = np.random.multivariate_normal(mu2, cov2)
        data.append([pt[0], pt[1], 1])

# 3. Export to .dat file for pgfplots
# We use space separation which is native to pgfplots
df = pd.DataFrame(data, columns=['x', 'y', 'label'])
df.to_csv('svm_data.dat', sep=' ', index=False)

print("Data saved to svm_data.dat")
print(df.head()) # Preview
exit()

# --- Part (b) Dual SVM Optimization (Manual) ---
train_n = 600
X_train, y_train = X[:train_n], labels[:train_n]
X_test, y_test = X[train_n:], labels[train_n:]

# Dual Variables alpha
alpha = cp.Variable(train_n)
# Kernel Matrix (Linear Kernel)
K = X_train @ X_train.T
# Quadratic term: 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K_ij)
# Vectorized: 0.5 * alpha^T * diag(y) * K * diag(y) * alpha
Y_diag = np.diag(y_train)
P = Y_diag @ K @ Y_diag
# Objective: Maximize sum(alpha) - 0.5 * quad_form
obj = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.psd_wrap(P)))

# Constraints: 0 <= alpha <= C, sum(alpha * y) == 0
C = 1.0 # Standard Soft Margin Parameter
constraints = [alpha >= 0, alpha <= C, alpha @ y_train == 0]

prob = cp.Problem(obj, constraints)
prob.solve()
alpha_val = alpha.value

# Retrieve weights w and bias b
w = np.sum((alpha_val[:, None] * y_train[:, None]) * X_train, axis=0)
# Support vectors (where 0 < alpha < C approx)
sv_idx = np.where((alpha_val > 1e-4) & (alpha_val < C - 1e-4))[0]
if len(sv_idx) > 0:
    b = np.mean(y_train[sv_idx] - X_train[sv_idx] @ w)
else:
    b = 0 # Fallback

# Plotting Decision Boundary
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], c='blue', marker='s')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', marker='x')

# Create grid to plot line
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = -(w[0] * xx + b) / w[1]
plt.plot(xx, yy, 'k-', label='Decision Boundary')
plt.title(f'SVM Dual Solution (Train n={train_n})')
plt.legend()
plt.show()

# --- Part (c) Classification on Holdout ---
preds = np.sign(X_test @ w + b)
error_rate = np.mean(preds != y_test)
print(f"Manual SVM Error Rate on Holdout: {error_rate:.4f}")

# Plot Holdout
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], c='blue', marker='s', alpha=0.3, label='True -1')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', marker='x', alpha=0.3, label='True 1')
# Highlight errors
errors = X_test[preds != y_test]
plt.scatter(errors[:, 0], errors[:, 1], facecolors='none', edgecolors='green', s=100, label='Errors')
plt.plot(xx, yy, 'k-')
plt.title(f'SVM Classification on Test Data (Error: {error_rate:.2%})')
plt.legend()
plt.show()

# --- Part (d) Comparison with Sklearn ---
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)
sklearn_acc = clf.score(X_test, y_test)
print(f"Sklearn Error Rate: {1 - sklearn_acc:.4f}")