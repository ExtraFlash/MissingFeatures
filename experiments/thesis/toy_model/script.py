import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def main():
    train_ratio = 0.8
    n_samples = 100000

    # Generate synthetic data
    first_feature = torch.randn(n_samples, 1) + 2
    second_feature = first_feature + torch.randn(n_samples, 1) * 0.1

    # Plot distributions of the two features
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Input Feature Distributions', fontsize=16, fontweight='bold', y=0.95)

    features = [
        (first_feature.detach().numpy().flatten(), r'$x^1$', '#1f77b4'),
        (second_feature.detach().numpy().flatten(), r'$x^2$', '#ff7f0e')
    ]

    for ax, (data, label, color) in zip(axes, features):
        ax.hist(data, bins=50, alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Distribution of {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.02, 0.98, f'μ = {mean_val:.3f}\nσ = {std_val:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("feature_distributions.png")

    x = torch.cat((first_feature, second_feature), dim=1)
    y = first_feature + second_feature + torch.randn(n_samples, 1) * 0.1

    # split into train val
    n_train = int(n_samples * train_ratio)
    x_train = x[:n_train]
    x_val = x[n_train:]
    y_train = y[:n_train]
    y_val = y[n_train:]

    # Define a simple linear regression model
    model_full = nn.Linear(2, 1, bias=True)

    train_loss_full = []
    val_loss_full = []

    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_full.parameters(), lr=0.01)
    for epoch in range(500):
        model_full.train()
        optimizer.zero_grad()
        outputs = model_full(x_train)
        loss_train = criterion(outputs, y_train)
        loss_train.backward()
        optimizer.step()

        # Evaluate the model with both features (full model)
        model_full.eval()
        with torch.no_grad():
            y_pred_full = model_full(x_val)
            loss_val = criterion(y_pred_full, y_val)

        print(f"Epoch [{epoch + 1}/100], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")
        train_loss_full.append(loss_train.item())
        val_loss_full.append(loss_val.item())

    # y_train_preds = model_full(x_train)
    y_val_preds = model_full(x_val)

    # Get the trained parameters
    weight = model_full.weight.data  # shape: [1, 2]
    bias = model_full.bias.data  # shape: [1]

    print(f"Learned weights: {weight}")
    print(f"Learned bias: {bias}")

    # Evaluate with only the first feature
    # This simulates the scenario where x2 is not available at inference
    x_val_first_only = x_val[:, 0:1]  # Only first feature

    # Create a model that only uses the first feature
    # Using the learned weight for the first feature and the bias
    with torch.no_grad():
        # Manual prediction: theta1 * x1 + b (ignoring theta2 * x2)
        y_pred_first_only = weight[0, 0] * x_val_first_only + bias
        loss_first_only = criterion(y_pred_first_only, y_val)
        print(f"First feature only validation loss: {loss_first_only.item():.4f}")

    ###### Train only on first feature ######

    model_first = nn.Linear(1, 1, bias=True)
    train_loss_first = []
    val_loss_first = []

    x_first = x[:, 0:1]  # Only first feature
    x_train_first = x_first[:n_train]
    x_val_first = x_first[n_train:]

    optimizer = torch.optim.Adam(model_first.parameters(), lr=0.01)

    # Train the model
    for epoch in range(500):
        model_first.train()
        optimizer.zero_grad()
        outputs = model_first(x_train_first)
        loss_train = criterion(outputs, y_train)
        loss_train.backward()
        optimizer.step()

        # Evaluate the model with only the first feature
        model_first.eval()
        with torch.no_grad():
            y_pred_first = model_first(x_val_first)
            loss_val = criterion(y_pred_first, y_val)

        print(f"Epoch [{epoch + 1}/300], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")
        train_loss_first.append(loss_train.item())
        val_loss_first.append(loss_val.item())

    # Get the trained parameters
    weight = model_first.weight.data  # shape: [1, 2]
    # bias = model_first.bias.data  # shape: [1]

    print(f"Learned weights first feature: {weight}")
    # print(f"Learned bias first feature: {bias}")

    y_trained_on_first_preds = model_first(x_val_first)

    y_val_np = y_val.detach().numpy().flatten()
    y_val_preds_np = y_val_preds.detach().numpy().flatten()
    y_pred_first_only_np = y_pred_first_only.detach().numpy().flatten()
    y_trained_on_first_preds_np = y_trained_on_first_preds.detach().numpy().flatten()

    # Set up consistent bins for all plots
    all_data = np.concatenate([y_val_np, y_val_preds_np, y_pred_first_only_np, y_trained_on_first_preds_np])
    bins = np.linspace(all_data.min(), all_data.max(), 30)

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison: Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Data, colors, and titles for each subplot
    plot_data = [
        (y_val_np, '#1f77b4', 'True Values'),
        (y_val_preds_np, '#ff7f0e', 'Full Model Predictions'),
        (y_pred_first_only_np, '#2ca02c', 'Inference on First Feature Only'),
        (y_trained_on_first_preds_np, '#d62728', 'Model Trained on First Feature Only')
    ]

    # Subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Create each subplot
    for i, ((data, color, title), label) in enumerate(zip(plot_data, subplot_labels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Create histogram
        ax.hist(data, bins=bins, alpha=0.8, color=color, edgecolor='black', linewidth=0.5)

        # Customize each subplot
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Values', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add statistics text box
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.02, 0.98, f'μ = {mean_val:.3f}\nσ = {std_val:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add subplot label (a), (b), etc.
        ax.text(-0.05, 1.1, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Uncomment if needed for spacing

    # Save the figure
    plt.savefig("distribution_full_model.png")




if __name__ == '__main__':
    main()
