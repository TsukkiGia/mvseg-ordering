from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets


class RegressionModel:
    """Linear model with optional ridge (L2) regularization."""

    def __init__(
        self,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        *,
        n_features: Optional[int] = None,
        lr: float = 5e-2,
        ridge_lambda: float = 1e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        if float(ridge_lambda) < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}.")

        self.device = torch.device(device)
        self.lr = float(lr)
        self.ridge_lambda = float(ridge_lambda)
        self.x_train: Optional[torch.Tensor]
        self.y_train: Optional[torch.Tensor]

        if x_train is None and y_train is None and n_features is None:
            x_train, y_train = datasets.make_regression(
                n_samples=100,
                n_features=1,
                random_state=0,
            )

        if (x_train is None) != (y_train is None):
            raise ValueError("x_train and y_train must both be provided or both be None.")

        if x_train is None and y_train is None:
            if n_features is None or int(n_features) < 1:
                raise ValueError("n_features must be provided and >= 1 when x_train/y_train are omitted.")
            input_size = int(n_features)
            self.x_train = None
            self.y_train = None
        else:
            x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
            y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(1)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)
            if x_tensor.ndim != 2:
                raise ValueError(f"x_train must have shape [N, D], got {tuple(x_tensor.shape)}.")
            if y_tensor.ndim != 2 or int(y_tensor.shape[1]) != 1:
                raise ValueError(f"y_train must have shape [N] or [N, 1], got {tuple(y_tensor.shape)}.")
            if int(x_tensor.shape[0]) != int(y_tensor.shape[0]):
                raise ValueError(
                    f"x_train and y_train must have the same rows, got {x_tensor.shape[0]} and {y_tensor.shape[0]}."
                )

            input_size = int(x_tensor.shape[1])
            self.x_train = x_tensor
            self.y_train = y_tensor

        self.model = nn.Linear(input_size, 1).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def _ridge_penalty(self) -> torch.Tensor:
        return self.model.weight.pow(2).sum()

    def fit(self, num_epochs: int = 100, verbose: bool = True) -> list[float]:
        if self.x_train is None or self.y_train is None:
            raise ValueError("fit requires x_train and y_train.")

        losses: list[float] = []
        for epoch in range(int(num_epochs)):
            y_pred = self.model(self.x_train)
            mse_loss = self.loss_fn(y_pred, self.y_train)
            ridge_loss = self.ridge_lambda * self._ridge_penalty()
            loss = mse_loss + ridge_loss
            losses.append(float(loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                print(
                    f"epoch={epoch + 1:04d} "
                    f"mse={float(mse_loss.item()):.6f} "
                    f"ridge={float(ridge_loss.item()):.6f} "
                    f"total={float(loss.item()):.6f}"
                )

        return losses

    # Backward-compatible alias used by older code.
    def train(self, num_epochs: int = 100, plot_path: Optional[str | Path] = None) -> list[float]:
        losses = self.fit(num_epochs=num_epochs, verbose=True)
        if plot_path is not None and self.x_train is not None and self.y_train is not None:
            x_np = self.x_train.detach().cpu().numpy()
            y_np = self.y_train.detach().cpu().numpy().reshape(-1)
            pred_np = self.predict(x_np)
            if x_np.shape[1] == 1:
                sort_idx = np.argsort(x_np[:, 0])
                path = Path(plot_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                plt.plot(x_np[sort_idx, 0], y_np[sort_idx], "ro")
                plt.plot(x_np[sort_idx, 0], pred_np[sort_idx], "b")
                plt.savefig(path, dpi=200, bbox_inches="tight")
                plt.close()
        return losses

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        if x_tensor.ndim != 2:
            raise ValueError(f"x must have shape [N, D], got {tuple(x_tensor.shape)}.")
        with torch.no_grad():
            y_pred = self.model(x_tensor).detach().cpu().numpy().reshape(-1)
        return y_pred

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "lr": self.lr,
            "ridge_lambda": self.ridge_lambda,
            "n_features": int(self.model.in_features),
        }
        torch.save(payload, out_path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "RegressionModel":
        payload = torch.load(Path(path), map_location=torch.device(device))
        model = cls(
            x_train=None,
            y_train=None,
            n_features=int(payload["n_features"]),
            lr=float(payload.get("lr", 5e-2)),
            ridge_lambda=float(payload.get("ridge_lambda", 0.0)),
            device=device,
        )
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model


if __name__ == "__main__":
    demo_model = RegressionModel()
    demo_model.train(num_epochs=100, plot_path="figures/regression/regression_fit.png")
