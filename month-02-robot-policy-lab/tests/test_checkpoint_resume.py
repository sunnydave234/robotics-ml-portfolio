"""
Run A: 100 steps, uninterrupted.
Run B: 50 steps -> save -> rebuild from scratch -> load -> 50 more steps.
        Must match Run A exactly.
Run C: same as B but skips RNG restore. Must NOT match Run A - this is
        the check that proves the test itself is meaningful (see W2D4
        notes: the un-modified harness model has no RNG-consuming layer,
        so without Dropout below, this negative check would never fire).

Run: PYTHONPATH=$(pwd) python tests/test_checkpoint_resume.py
"""
import torch
import torch.nn as nn

from robot_policy_lab.utils.checkpoint import load_checkpoint, save_checkpoint
from robot_policy_lab.utils.reproducibility import seed_everything

DEVICE ="mps" if torch.backends.mps.is_available() else "cpu"
CKPT_PATH = "outputs/test_checkpoint_step50.pt"
SEED, TOTAL_STEPS, SPLIT = 42, 100, 50

def _make_model_and_data(seed: int, n: int):
    seed_everything(seed)
    # Dropout added vs. the W1D5 harness — it randomly zeroes 10% of
    # activations every forward pass DURING TRAINING, which forces a real
    # draw from the RNG. This stands in for ACT's CVAE noise sampling.
    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(64, 2)
    ).to(DEVICE)
    model.train()   # Dropout only activates in train model - explicit on purpose

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    torch.manual_seed(seed)
    batches = [torch.randn(32, 2, device=DEVICE) for _ in range(n)]
    return model, optimizer, batches


def _train(model, optimizer, batches) -> list[float]:
    losses = []
    for x in batches:
        pred = model(x)
        loss = nn.functional.mse_loss(pred, torch.zeros_like(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 8))
    return losses


def run_uninterrupted() -> list[float]:
    model, opt, batches = _make_model_and_data(SEED, TOTAL_STEPS)
    return _train(model, opt, batches)


def run_resumed(restore_rng: bool) -> list[float]:
    model, opt, batches = _make_model_and_data(SEED, TOTAL_STEPS)
    losses_pre = _train(model, opt, batches[:SPLIT])
    save_checkpoint(CKPT_PATH, step=SPLIT, model=model, optimizer=opt)

    # Pretend this is a brand new process: rebuild model/optimizer/data
    # from nothing, then overwrite with the saved checkpoint.
    model2, opt2, batches2 = _make_model_and_data(SEED, TOTAL_STEPS)
    load_checkpoint(CKPT_PATH, model=model2, optimizer=opt2, restore_rng=restore_rng)
    losses_post = _train(model2, opt2, batches2[SPLIT:])
    return losses_pre + losses_post


if __name__ == "__main__":
    ref = run_uninterrupted()

    resumed = run_resumed(restore_rng=True)
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(ref, resumed)) if a != b]
    assert not mismatches, f"Resume NOT identical, first mismatch: {mismatches[0]}"
    print(f"Resume identical ✓   steps 51-100 float-equal ({len(ref)} steps total)")

    broken = run_resumed(restore_rng=False)
    post_split_diffs = [i for i, (a, b) in enumerate(zip(ref, broken)) if i >= SPLIT and a != b]
    assert post_split_diffs, "RNG-skip did not change anything — test is vacuous."
    print(f"Negative control ✓   RNG-skip diverges at step {post_split_diffs[0] + 1}")
    print(f"    Device: {DEVICE}")