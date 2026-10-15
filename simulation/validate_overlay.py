import cv2
import json
import math
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Create BEV overlay validation video')
    parser.add_argument('--video-input', '-i', required=True, help='Input video path')
    parser.add_argument('--video-output', '-o', required=True, help='Output video path')
    parser.add_argument('--json-input', '-j', required=True, help='Pre-filtered JSON data path')
    parser.add_argument('--target-key', '-t', default='npc',
                        help='Target actor key: npc or walker')
    return parser.parse_args()


def calculate_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


def get_global_bounds(branch_data, target_key):
    """Scan all JSON data to find coordinate bounds (Bounding Box) for the entire path."""
    xs, ys = [], []
    for step in branch_data:
        xs.append(step['ego']['x'])
        ys.append(step['ego']['y'])
        if target_key in step:
            xs.append(step[target_key]['x'])
            ys.append(step[target_key]['y'])

    pad = 15.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_hud(frame, data, target_key):
    """Draw Text Telemetry panel (Left side)."""
    overlay = frame.copy()
    x, y, w, h = 20, 20, 350, 220
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    t = data['time_sec']
    ego = data['ego']
    target = data.get(target_key, {})
    dist = calculate_distance(ego, target) if target else 0.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    alert_color = (0, 0, 255) if ego.get('ax', 0) < -3.0 else (0, 255, 0)

    cv2.putText(frame, f"TIME: {t:.1f} s", (x+15, y+35), font, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"--- EGO VEHICLE ---", (x+15, y+70), font, 0.6, color, 1)
    cv2.putText(frame, f"Speed: {ego.get('speed_kmh', 0):.1f} km/h",
                (x+15, y+95), font, 0.6, color, 1)
    cv2.putText(frame, f"Accel: {ego.get('ax', 0):.1f} m/s^2",
                (x+15, y+120), font, 0.6, alert_color, 2)

    cv2.putText(frame, f"--- {target_key.upper()} ---", (x+15, y+155), font, 0.6, color, 1)
    cv2.putText(frame, f"Speed: {target.get('speed_kmh', target.get('speed', 0)):.1f} km/h",
                (x+15, y+180), font, 0.6, color, 1)

    dist_color = (0, 0, 255) if dist < 10.0 else color
    cv2.putText(frame, f"Distance: {dist:.1f} m", (x+15, y+205), font, 0.6, dist_color, 2)

    if data.get("collision", False):
        cv2.putText(frame, "!!! COLLISION DETECTED !!!", (x+15, y+250),
                    font, 0.8, (0, 0, 255), 3)

    return frame


def draw_global_bev(frame, data, bounds, history, target_key):
    """Draw Global BEV with Coordinates and Trajectory (Right side)."""
    min_x, max_x, min_y, max_y = bounds
    bev_w, bev_h = 350, 450

    bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    bev_map[:] = (30, 30, 40)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    scale = min(bev_h / range_x, bev_w / range_y)

    cx_offset = (bev_w - range_y * scale) / 2
    cy_offset = (bev_h - range_x * scale) / 2

    def to_px(wx, wy):
        px = int(cx_offset + (wy - min_y) * scale)
        py = int(bev_h - (cy_offset + (wx - min_x) * scale))
        return px, py

    # Grid
    grid_step = max(1, int(10 * scale))
    for i in range(0, max(bev_w, bev_h), grid_step):
        cv2.line(bev_map, (i, 0), (i, bev_h), (60, 60, 70), 1)
        cv2.line(bev_map, (0, i), (bev_w, i), (60, 60, 70), 1)

    # Trajectory tails
    for i in range(1, len(history['ego'])):
        p1 = to_px(*history['ego'][i-1])
        p2 = to_px(*history['ego'][i])
        cv2.line(bev_map, p1, p2, (255, 150, 0), 2, cv2.LINE_AA)

    for i in range(1, len(history['target'])):
        p1 = to_px(*history['target'][i-1])
        p2 = to_px(*history['target'][i])
        target_color = (0, 0, 255) if target_key == "npc" else (0, 255, 255)
        cv2.line(bev_map, p1, p2, target_color, 2, cv2.LINE_AA)

    # Current positions
    ego = data['ego']
    target = data.get(target_key, {})

    ego_px, ego_py = to_px(ego['x'], ego['y'])
    cv2.circle(bev_map, (ego_px, ego_py), 6, (255, 200, 0), -1)
    cv2.putText(bev_map, f"EGO({ego['x']:.1f}, {ego['y']:.1f})",
                (ego_px + 10, ego_py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

    if target:
        tgt_px, tgt_py = to_px(target['x'], target['y'])
        tgt_color = (0, 0, 255) if target_key == "npc" else (0, 255, 255)
        cv2.circle(bev_map, (tgt_px, tgt_py), 6, tgt_color, -1)
        cv2.putText(bev_map, f"{target_key.upper()}({target['x']:.1f}, {target['y']:.1f})",
                    (tgt_px + 10, tgt_py + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, tgt_color, 1, cv2.LINE_AA)

    cv2.rectangle(bev_map, (0, 0), (bev_w-1, bev_h-1), (255, 255, 255), 2)

    h, w = frame.shape[:2]
    margin = 20
    # Check if the frame is large enough to contain the BEV
    if h >= margin + bev_h and w >= margin + bev_w + margin:
        frame[margin:margin+bev_h, w-bev_w-margin:w-margin] = bev_map
        cv2.putText(frame, "Global BEV Trajectory", (w-bev_w-margin, margin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def main():
    args = parse_args()

    print(f"  Rendering BEV overlay...")
    with open(args.json_input, 'r') as f:
        branch_data = json.load(f)

    if not branch_data:
        print("  [WARN] JSON is empty, skipping validate overlay.")
        return

    bounds = get_global_bounds(branch_data, args.target_key)

    cap = cv2.VideoCapture(args.video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output, fourcc, fps, (width, height))

    frame_idx = 0
    history = {'ego': [], 'target': []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(branch_data):
            json_step = branch_data[frame_idx]

            history['ego'].append((json_step['ego']['x'], json_step['ego']['y']))
            if args.target_key in json_step:
                history['target'].append(
                    (json_step[args.target_key]['x'], json_step[args.target_key]['y']))

            frame = draw_hud(frame, json_step, args.target_key)
            frame = draw_global_bev(frame, json_step, bounds, history, args.target_key)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  Rendered {frame_idx} frames → {args.video_output}")


if __name__ == '__main__':
    main()