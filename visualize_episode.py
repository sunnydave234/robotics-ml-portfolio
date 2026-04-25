import cv2
import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from extract_episode import extract_episode


def tensor_to_bgr_frame(frame_tensor: torch.Tensor, upscale: int = 4) -> np.ndarray:
    """
    Convert a single (C, H, W) float32 RGB frame tensor to a (H*upscale, W*upscale, 3) uint8 BGR numpy array.
    This is the single conversion boundary in the file.
    This is the only place in the file where tensors become numpy arrays.

    Args:
        frame_tensor: shape (C, H, W), float32 in [0.0, 1.0], CPU
        upscale:      integer scale factor (default 4 → 96px becomes 384px)

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
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    if upscale > 1:
        H, W = frame_bgr.shape[:2]
        # INTER_NEAREST preserves crisp pixel art look; use INTER_LINEAR for smooth
        frame_bgr = cv2.resize(
            frame_bgr, (W * upscale, H * upscale),
            interpolation=cv2.INTER_NEAREST,
        )

    return frame_bgr


def write_episode_video_with_actions(episode: dict, output_path: str, fps: int = 10, upscale: int = 4) -> None:
    """
    Write a rendered episode MP4 with action overlays on each frame.

    Args:
        episode: dict from extract_episode() — keys: images (T,C,H,W), actions (T,action_dim)
        output_path: destination .mp4 path
        fps: playback frame rate
        upscale: integer upscale factor applied in tensor_to_bgr_frame
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
    out_H, out_W = H * upscale, W * upscale

    print(f"Writing {T} frames @ {out_W}x{out_H} (upscaled {upscale}x) → {output_path}")

    # mp4v is the codec that works reliably on M4 Mac with stock OpenCV
    # avc1 (H.264) gives better compression but requires a licensed OpenCV build
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # NOTE: VideoWriter takes (width, height) - opposite of tensor shape convention (H, W)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_W, out_H))

    # First silent failure point - check before the loop, not after 200 frames
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed to open. HCeck path exists and codec is valid.\n"
            f"Path: {output_path}"
        )

    for i in range(T):
        # images[i] is still (C,H,W) tensor here
        # all conversion happens inside tensor_to_bgr_frame
        bgr = tensor_to_bgr_frame(images[i], upscale=upscale)          # tensor → numpy at the boundary
        bgr = add_action_overlay(bgr, actions[i], step=i)  # actions[i] stays tensor until .item()
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
        bgr_frame: uint8 numpy array (H, W, 3) — already upscaled
        action: 1D tensor shape (action_dim,) — for pusht this is (2,): [x, y] target position
        step: frame index displayed top-right corner

    Returns:
        Annotated copy of the frame — original not mutated
    """
    frame = bgr_frame.copy()    # don't mutate — caller may need the original

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.5
    thickness = 1
    white     = (255, 255, 255)
    black     = (0, 0, 0)       # shadow — makes text readable on any background color
    
    # Compute line height dynamically — don't hardcode pixel values
    (_, line_h), baseline = cv2.getTextSize("Ag", font, scale, thickness)
    line_h += baseline + 2

    # Step counter — top-right corner
    step_text = f"step:{step}"
    (tw, _), _ = cv2.getTextSize(step_text, font, scale, thickness)
    x = frame.shape[1] - tw - 8
    cv2.putText(frame, step_text, (x + 1, line_h),     font, scale, black, thickness + 1)
    cv2.putText(frame, step_text, (x,     line_h),     font, scale, white, thickness)

    # Action values — left side, one line per dimension
    # action is still a tensor here — .item() extracts each scalar without numpy conversion
    for i in range(action.shape[0]):
        val  = action[i].item()           # tensor scalar → Python float, device-agnostic
        text = f"a[{i}]: {val:+.3f}"
        y = (i + 1) * line_h
        cv2.putText(frame, text, (6, y), font, scale, black, thickness + 1)
        cv2.putText(frame, text, (5, y), font, scale, white, thickness)

    return frame


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a LeRobot episode as an MP4 with action overlays."
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to render (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output MP4 path. Defaults to outputs/episode_<N>.mp4"
    )
    parser.add_argument(
        "--dataset", type=str, default="lerobot/pusht",
        help="HuggingFace dataset repo ID (default: lerobot/pusht)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Playback frame rate (default: 10)"
    )
    parser.add_argument(
        "--upscale", type=int, default=4,
        help="Integer upscale factor for small source frames (default: 4, 96->384px)"
    )
    args = parser.parse_args()

    output = args.output or f"outputs/episode_{args.episode:03d}.mp4"

    print(f"Loading {args.dataset}...")
    ds = LeRobotDataset(args.dataset)

    print(f"Extractin episode {args.episode} / {ds.meta.total_episodes - 1}...")
    ep = extract_episode(ds, episode_idx=args.episode)

    write_episode_video_with_actions(
        episode=ep,
        output_path=output,
        fps=args.fps,
        upscale=args.upscale,
    )
    print(f"\nDone -> {output}")
