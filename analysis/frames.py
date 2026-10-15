"""Frame-selection helper for non-protocol datasets.

Protocol datasets bypass this helper and pass their exact eight-frame window.
It serves datasets without a manifest and isolated unit tests.
"""

from typing import List


def select_frames(frame_paths: List[str], num_frames: int) -> List[str]:
    """Uniformly sample `num_frames` items from `frame_paths`, endpoints included
    and order preserved. Returns all frames unchanged when there are fewer than
    requested (no duplication); the middle frame when num_frames <= 1."""
    n = len(frame_paths)
    if n == 0:
        return []
    if num_frames <= 1:
        return [frame_paths[n // 2]]
    if n <= num_frames:
        return list(frame_paths)
    idxs = [round(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    # Dedup while preserving order (rounding can collide for tiny n).
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(frame_paths[i])
    return out
