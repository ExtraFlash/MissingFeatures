import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Parameters
N = 50000            # Number of samples
p_x = 0.5           # Bernoulli parameter for x
num_features = 20    # 5 Bernoulli features per sample
# true_W = np.array([2.0, -1.0, 0.5, 1.5, -0.5, 2.0, -1.0, 0.5, 1.5, -0.5])  # True weights
true_W = np.random.normal(0, 1, num_features)
true_b = -1.0       # True bias

# Seed for reproducibility
# np.random.seed(42)

# Generate x from Bernoulli(p_x)
# x = np.random.binomial(1, p_x, (N, num_features)).astype(np.float32)

# generate x from normal distribution
x = np.random.normal(0, 1, (N, num_features)).astype(np.float32)

# Generate noise epsilon ~ N(0, 1)
epsilon = np.random.normal(0, 1, N).astype(np.float32)

# Compute z = W.T * x + b + epsilon
z = np.dot(x, true_W) + true_b + epsilon

# Compute p(y=1) = sigmoid(z)
p_y = 1 / (1 + np.exp(-z))

# Sample y from Bernoulli(p_y)
y = np.random.binomial(1, p_y).astype(np.float32)

# Convert to tensors
x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y).unsqueeze(1)  # Ensure y is a column vector


# Plot the histogram of p_y
plt.hist(p_y, bins=30, alpha=0.7, label='p(y=1)')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Histogram of p(y=1)')
plt.legend()
plt.show()

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=5):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Adjust for 5 input features

    def forward(self, x):
        logits = self.linear(x)
        y_pred = torch.sigmoid(logits)
        return y_pred


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=5):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Adjust for 5 input features

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def print_parameters(self):
        # Access weights and bias from the linear layer
        weight = self.linear.weight.data  # Shape: (1, input_dim)
        bias = self.linear.bias.data  # Shape: (1,)
        print(f'-----')
        print('Linear Model')
        print(f"Weights (w): {weight}")
        print(f"Bias (b): {bias}")


# Create a TensorDataset and DataLoader
dataset = TensorDataset(x_tensor, y_tensor)
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, criterion, optimizer, loader, num_epochs=100):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        average_loss = total_loss / len(loader.dataset)
        epoch_losses.append(average_loss)
    return epoch_losses


# Logistic Regression Model
logistic_model = LogisticRegressionModel(input_dim=num_features)
criterion_bce = nn.BCELoss()
optimizer_logistic = optim.SGD(logistic_model.parameters(), lr=0.1)

# Linear Regression Model
linear_model = LinearRegressionModel(input_dim=num_features)
criterion_mse = nn.MSELoss()
optimizer_linear = optim.SGD(linear_model.parameters(), lr=0.1)

# Train Logistic Regression Model
logistic_losses = train_model(logistic_model, criterion_bce, optimizer_logistic, loader)

# Train Linear Regression Model
linear_losses = train_model(linear_model, criterion_mse, optimizer_linear, loader)


def evaluate_model(model, x_data, y_true):
    model.eval()
    with torch.no_grad():
        outputs = model(x_data)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_true).float().mean().item()
        return accuracy


accuracy_logistic = evaluate_model(logistic_model, x_tensor, y_tensor)
print(f"Logistic Regression Model Accuracy: {accuracy_logistic * 100:.2f}%")

# Note: Even though it's a regression model, we threshold outputs for classification
accuracy_linear = evaluate_model(linear_model, x_tensor, y_tensor)
print(f"Linear Regression Model Accuracy: {accuracy_linear * 100:.2f}%")

# Compute ROC-AUC for both my_models
# Convert predicted probabilities to numpy arrays
logistic_pred = logistic_model(x_tensor).detach().numpy()
linear_pred = linear_model(x_tensor).detach().numpy()
auc_logistic = roc_auc_score(y_tensor, logistic_pred)
auc_linear = roc_auc_score(y_tensor, linear_pred)

real_auc = roc_auc_score(y, p_y)

print(f'Logistic Regression AUC: {auc_logistic:.4f}')
print(f'Linear Regression AUC: {auc_linear:.4f}')
print(f'Real AUC: {real_auc:.4f}')
linear_model.print_parameters()
