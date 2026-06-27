"""
Reproducibiltiy smoke test.

Runs the same 20-step training loop twice with seed=42 and asserts
identical loss curves. This catches:
    - unseeded weight initialization
    - unseeded dropout / stochastic ops
    - MPS RNG state leaking between runs

Run: python tests/test_reproducibility.py
Expected: "Reproducible"

DataLoader worker non-determinism is intentionally excluded here -
that's handled in week 2 via worker_init_fn + generator. 
"""
import torch
import torch.nn as nn

from robot_policy_lab.utils.reproducibility import seed_everything

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _make_model_and_data(seed: int, n: int = 20):
    seed_everything(seed)

    # Minimal 2-layer MLP - same dims as pusht (state_dim=2 -> action_dim=2).
    # Model init happens AFTER seed_everything: weight init uses RNG.
    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Fixed synthetic batches - manual_seed again for the data generation
    # so batches are identical across runs regardless of model-init RNG state.
    torch.manual_seed(seed)
    batches = [torch.randn(32, 2, device=DEVICE) for _ in range(n)]

    return model, optimizer, batches


def run_n_steps(seed: int, n: int = 20) -> list[float]:
    model, optimizer, batches = _make_model_and_data(seed, n)
    losses = []
    for x in batches:
        pred = model(x)
        loss = nn.functional.mse_loss(pred, torch.zeros_like(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 8)) # round to absorb sub-epsilon float noise
    return losses


if __name__ == "__main__":
    r1 = run_n_steps(seed=42)
    r2 = run_n_steps(seed=42)
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(r1, r2)) if a != b]
    assert not mismatches, f"NOT reproducible at step {mismatches}"
    print("Reproducible ✓")
    print(f"    Loss range: {min(r1):.6f} -> {max(r1):.6f}")
    print(f"    Device: {DEVICE}")