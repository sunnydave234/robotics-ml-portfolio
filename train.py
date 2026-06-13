"""
train.py — W4D3
Trivial behavior-cloning MLP smoke test, with W&B logging.

Pipeline: RobotEpisodeDataset -> BCDataset (adds prev_action) -> DataLoader
Model:    BCMLP  (state + prev_action) -> action
Goal:     loss decreases over 10 epochs. That's it.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from config import OUTPUTS_DIR, checkpoint_path
from robot_dataset import RobotEpisodeDataset
from bc_dataset import BCDataset
from model import BCMLP

# Hyperparameters
STATE_DIM    = 2
ACTION_DIM   = 2
HIDDEN_DIM   = 256
BATCH_SIZE   = 64
EPOCHS       = 10
LR           = 1e-3
VAL_FRACTION = 0.1
SEED         = 42

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    base_ds = RobotEpisodeDataset(
        OUTPUTS_DIR / "metadata.parquet",
        normalize=True,     # z-score state/action via dataset_stats.json - MLPs train
                            # poorly on raw pusht coordinates (roughly 0-512 range)
        context_window=1,   # default - we never toruch "image", just keeping it explicit
    )
    full_ds = BCDataset(base_ds)

    n_val = int(len(full_ds) * VAL_FRACTION)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"train: {n_train}    val: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    # Model
    # Instantiate the model
    model = BCMLP(
        input_dim=STATE_DIM + ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=ACTION_DIM,
    ).to(device)

    # Create an Adam optimizer.
    # we pass model.parameters() to tell it which tensors to manage.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # We'll also grab a pre-built loss function from torch.nn
    loss_fn = nn.MSELoss() # Mean Squared Error Loss

    # W&B
    wandb.init(
        project="robot-data-forge",
        name="w4d3-bc-mlp-smoketest",
        config={
            "model": "BCMLP",
            "input_dim": STATE_DIM + ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": ACTION_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "optimizer": "Adam",
            "loss": "MSE",
            "dataset": "lerobot/pusht",
            "normalize": True,
            "n_train": n_train,
            "n_val": n_val,
        },
    )

    ckpt_path = checkpoint_path("bc_mlp_best.pt")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for batch in train_loader:
            inputs = torch.cat([batch["state"], batch["prev_action"]], dim=-1).to(device)
            targets = batch["action"].to(device) # y_true

            ### Forward Pass ###
            preds = model(inputs)   # y_hat
            ### Calculate Loss ###
            loss = loss_fn(preds, targets)

            ### The Three-Line Mantra ###
            # 1. Zero the gradients
            optimizer.zero_grad()
            # 2. Compute gradients
            loss.backward()
            # 3. Update the parameters
            optimizer.step()

            wandb.log({"train/loss": loss.item()})
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.cat([batch["state"], batch["prev_action"]], dim=-1).to(device)
                targets = batch["action"].to(device)
                val_losses.append(loss_fn(model(inputs), targets).item())

        val_loss = sum(val_losses) / len(val_losses)
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{EPOCHS}  val/loss={val_loss:.6f}")

        # ── Checkpoint ──────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_kwargs": {
                    "input_dim": STATE_DIM + ACTION_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "output_dim": ACTION_DIM,
                },
            }, ckpt_path)

    # ── Log best checkpoint as a W&B artifact ─────────────────────────────
    artifact = wandb.Artifact("bc_mlp_best", type="model")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
