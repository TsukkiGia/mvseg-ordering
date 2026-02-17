import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from pathlib import Path


class RegressionModel():
    def __init__(self, x_train=None, y_train=None, lr=5e-2):
        if x_train is None and y_train is None:
            x_train, y_train = datasets.make_regression(n_samples=100, n_features=1)
        elif x_train is None or y_train is None:
            raise ValueError("x_train and y_train must both be provided or both be None.")

        self.x_train = torch.as_tensor(x_train, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(
                f"x_train and y_train must have the same number of rows, got "
                f"{self.x_train.shape[0]} and {self.y_train.shape[0]}."
            )

        _, n_features = self.x_train.shape
        input_size = n_features
        output_size = 1

        # define model
        self.model = nn.Linear(input_size, output_size)

        # define loss and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, num_epochs=100, plot_path="figures/regression/regression_fit.png"):
        losses = []
        for epoch in range(num_epochs):
            y_pred = self.model(self.x_train)
            loss = self.loss_fn(y_pred, self.y_train)
            losses.append(loss.item())
            print(f"epoch: {epoch + 1}, loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        predicted = self.model(self.x_train).detach().cpu().numpy().reshape(-1)
        x_np = self.x_train.detach().cpu().numpy().reshape(-1)
        y_np = self.y_train.detach().cpu().numpy().reshape(-1)
        sort_idx = np.argsort(x_np)

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.plot(x_np[sort_idx], y_np[sort_idx], "ro")
        plt.plot(x_np[sort_idx], predicted[sort_idx], "b")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    model = RegressionModel()
    model.train()
