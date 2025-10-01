import math
from pathlib import Path

import numpy as np
import imageio
import torch


def save_video_grid(videos, out_path, fps: int = 5, grid_size: tuple[int, int] | None = None):
    """
    Save a batch of videos (or single-frame images) as a tiled video.

    Args:
        videos: Tensor or ndarray. Accepted shapes:
            - (N, T, C, H, W) : batch of videos
            - (N, C, H, W)    : batch of single-frame images (treated as T=1)
        out_path: str or Path to the output video file (e.g. .mp4).
        fps: frames per second.
        grid_size: (rows, cols). If None, will pick a near-square layout.

    Notes:
        - Input may be torch.Tensor (on any device) or numpy ndarray.
        - Input values are expected in [0,255] for uint8 or [0,1] for floats.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy on CPU
    if torch.is_tensor(videos):
        vids = videos.detach().cpu().numpy()
    else:
        vids = np.array(videos)

    # Normalize shapes: (N, T, C, H, W)
    if vids.ndim == 4:  # (N, C, H, W) -> treat as T=1
        vids = vids[:, None, ...]
    if vids.ndim != 5:
        raise ValueError(f"Unsupported videos shape: {vids.shape}. Expected 4D or 5D tensor.")

    N, T, C, H, W = vids.shape

    # Handle float values in [0,1]
    if np.issubdtype(vids.dtype, np.floating):
        vids = np.clip(vids * 255.0, 0, 255).astype(np.uint8)
    else:
        vids = vids.astype(np.uint8)

    # Ensure 3 channels for output
    if C == 1:
        vids = np.repeat(vids, 3, axis=2)
        C = 3
    elif C == 4:
        # Drop alpha if present
        vids = vids[:, :, :3, :, :]
        C = 3
    elif C != 3:
        raise ValueError(f"Unsupported channel count: {C}. Expected 1, 3 or 4.")

    # Determine grid size
    if grid_size is None:
        rows = int(math.ceil(math.sqrt(N)))
        cols = int(math.ceil(N / rows))
    else:
        rows, cols = grid_size
        if rows * cols < N:
            raise ValueError(f"grid_size {grid_size} too small for {N} videos")

    # Precompute canvas size
    canvas_h = rows * H
    canvas_w = cols * W

    # Build frames for each time step and write video
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
        for t in range(T):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for i in range(N):
                r = i // cols
                c = i % cols
                y0 = r * H
                x0 = c * W
                frame = vids[i, t].transpose(1, 2, 0)  # C,H,W -> H,W,C
                canvas[y0 : y0 + H, x0 : x0 + W, :] = frame
            writer.append_data(canvas)

    return str(out_path)