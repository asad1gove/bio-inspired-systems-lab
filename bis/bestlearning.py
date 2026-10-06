import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Dataset Generation & Setup
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, train_y = torch.FloatTensor(X_tr), torch.LongTensor(y_tr)
val_X, val_y = torch.FloatTensor(X_val), torch.LongTensor(y_val)

# 2. Model Architecture & Objective Function
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):
        return self.net(x)

def get_val_loss(lr):
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(15):
        optimizer.zero_grad()
        loss = criterion(model(train_X), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return criterion(model(val_X), val_y).item()

# 3. Particle Swarm Optimization (PSO)
num_particles, iterations, bounds = 5, 5, (0.001, 0.5)
positions = np.random.uniform(bounds[0], bounds[1], num_particles)
velocities = np.random.uniform(-0.05, 0.05, num_particles)
personal_bests = positions.copy()
personal_best_scores = np.array([get_val_loss(p) for p in positions])
global_best_pos = personal_bests[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

history = [global_best_score]

for i in range(iterations):
    for p in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[p] = 0.5 * velocities[p] + 1.5 * r1 * (personal_bests[p] - positions[p]) + 1.5 * r2 * (global_best_pos - positions[p])
        positions[p] = np.clip(positions[p] + velocities[p], bounds[0], bounds[1])
        score = get_val_loss(positions[p])
        if score < personal_best_scores[p]:
            personal_best_scores[p], personal_bests[p] = score, positions[p]
        if score < global_best_score:
            global_best_score, global_best_pos = score, positions[p]
    history.append(global_best_score)

print(f"Optimal Learning Rate: {global_best_pos:.5f} with Val Loss: {global_best_score:.5f}")

# 4. History Visualization
plt.figure(figsize=(8, 5))
plt.plot(history, marker='o', color='b')
plt.title("PSO Optimization History")
plt.xlabel("Iteration")
plt.ylabel("Best Validation Loss")
plt.grid(True)
plt.show()
