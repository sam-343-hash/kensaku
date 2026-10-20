"""Example training script. Runs on Kaggle; anything saved to /kaggle/working
is downloaded back automatically by kaggle_agent.py."""
import json
import os

import torch
import torch.nn as nn

OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Toy data: learn y = 3x + 1
x = torch.randn(2048, 1, device=device)
y = 3 * x + 1 + 0.05 * torch.randn_like(x)

model = nn.Linear(1, 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    loss = nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.5f}")

torch.save(model.state_dict(), f"{OUT}/model.pt")
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"final_loss": loss.item(), "device": device}, f)
print("Saved model.pt and metrics.json")
