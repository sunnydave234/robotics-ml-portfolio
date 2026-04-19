import cv2
import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from extract_episode import extract_episode


def tensor_to_bgr_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single (C, H, W) float32 RGB frame tensor to a (H, W, 3) uint8 BGR numpy array.
    This is the only place in the file where tensors become numpy arrays.

    Args:
        frame_tensor: shape (C, H, W), float32 in [0.0, 1.0], CPU

    Returns:
        numpy array: shape (H, W, 3), dtype uint8, BGR channel order
    """
    # extract_episode already guarantees CPU — but be explicit
    if frame_tensor.device.type != "cpu":
        frame_tensor = frame_tensor.cpu()
    
    # (C, H, W) → (H, W, C): PyTorch channel-first → OpenCV channel-last
    frame_tensor = frame_tensor.permute(1,2,0)

    # .numpy() requires contiguous memory — permute() breaks the contiguity guarantee
    frame_tensor = frame_tensor.contiguous()

    # Scale [0.0, 1.0] → [0, 255] and cast to uint8, entirely in PyTorch
    frame_tensor = frame_tensor.mul(255).clamp(0, 255).byte()

    # Zero-copy view — no data is duplicated
    frame_np = frame_tensor.numpy()

    # OpenCV expects BGR — LeRobot frames are RGB
    return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)


def write_episode_video_with_actions(episode: dict, output_path: str, fps: int = 10) -> None:
    """
    Write a rendered episode MP4 with action overlays on each frame.

    Args:
        episode: dict from extract_episode() — keys: images (T,C,H,W), actions (T,action_dim)
        output_path: destination .mp4 path
        fps: playback frame rate
    """
    images = episode['images']      #(T,C,H,W)
    actions = episode["actions"]   # (T, action_dim)

    # Catch mismatches before the loop — not on frame 47
    assert images.shape[0] == actions.shape[0], (
        f"Frame/action count mismatch: {images.shape[0]} frames vs {actions.shape[0]} actions"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T,C,H,W = images.shape
    print(f"Writing {T} frames @ {W}x{H} -> {output_path}")

    # mp4v is the codec that works reliably on M4 Mac with stock OpenCV
    # avc1 (H.264) gives better compression but requires a licensed OpenCV build
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # NOTE: VideoWriter takes (width, height) - opposite of tensor shape convention (H, W)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    # First silent failure point - check before the loop, not after 200 frames
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed to open. HCeck path exists and codec is valid.\n"
            f"Path: {output_path}"
        )

    for i in range(T):
        # images[i] is still (C,H,W) tensor here
        # all conversion happens inside tensor_to_bgr_frame
        bgr = tensor_to_bgr_frame(images[i])           # tensor → numpy at the boundary
        bgr = add_action_overlay(bgr, actions[i], i)   # actions[i] stays tensor until .item()
        writer.write(bgr)

    writer.release()

    # Second silent failure point - a valid MP4 is always multiple kilobytes
    file_size = output_path.stat().st_size
    print(f"    Size: {file_size / 1024:.1f} KB")
    if file_size < 1000:
        raise RuntimeError(
            f"Output suspiciously small ({file_size} bytes) - "
            f"codec likely failed silently. Check frame dtype and dimensions."
        )


def add_action_overlay(bgr_frame: np.ndarray, action: torch.Tensor, step: int) -> np.array:
    """
    Draw action vector values onto a BGR frame.

    Args:
        bgr_frame: uint8 numpy array (H, W, 3) — already converted by tensor_to_bgr_frame
        action: 1D tensor shape (action_dim,) — for pusht this is (2,): [x, y] target position
        step: frame index displayed top-right corner

    Returns:
        Annotated copy of the frame — original not mutated
    """
    frame = bgr_frame.copy()    # don't mutate — caller may need the original

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.25
    thickness = 1
    white     = (255, 255, 255)
    black     = (0, 0, 0)       # shadow — makes text readable on any background color
    line_h    = 11              # vertical spacing between lines, in pixels

    # Step counter — top-right corner
    step_text = f"step:{step}"
    x = frame.shape[1] - 55   # 100px from right edge
    cv2.putText(frame, step_text, (x + 1, 9), font, scale, black, thickness + 1)  # shadow
    cv2.putText(frame, step_text, (x,     8), font, scale, white, thickness)       # text

    # Action values — left side, one line per dimension
    # action is still a tensor here — .item() extracts each scalar without numpy conversion
    for i in range(action.shape[0]):
        val  = action[i].item()        # tensor scalar → Python float, device-agnostic
        text = f"a[{i}]: {val:+.1f}"  # +/- sign always shown, 3 decimal places

        y = 9 + i * line_h
        cv2.putText(frame, text, (6, y), font, scale, black, thickness + 1)  # shadow
        cv2.putText(frame, text, (5, y), font, scale, white, thickness)       # text

    return frame


if __name__ == "__main__":
    ds = LeRobotDataset("lerobot/pusht")

    for ep_idx in range(3):
        ep = extract_episode(ds, episode_idx=ep_idx)

        write_episode_video_with_actions(
            episode=ep,
            output_path=f"outputs/episode_{ep_idx:03d}.mp4",
            fps=10,
        )

    print("\nDone. Check outputs/")
