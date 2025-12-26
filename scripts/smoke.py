"""Minimal smoke test to validate src imports and a forward pass.

Run:
    python scripts/smoke.py
"""
import torch

from src.model import UNet


def main():
    model = UNet(in_channels=1, n_classes=4)
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 128, 128), f"Unexpected output shape: {y.shape}"
    print("Smoke test passed: forward() OK with shape", tuple(y.shape))


if __name__ == "__main__":
    main()
