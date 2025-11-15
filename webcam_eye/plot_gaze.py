import csv
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import pyautogui
except Exception:
    pyautogui = None


def read_csv(path):
    xs = []
    ys = []
    blinks = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = row.get('x')
            y = row.get('y')
            blink = int(row.get('blink') or 0)

            xs.append(float(x) if x != '' and x is not None else math.nan)
            ys.append(float(y) if y != '' and y is not None else math.nan)
            blinks.append(blink)

    return np.array(xs), np.array(ys), np.array(blinks)


def plot(path, out_image=None, show=True, overlay=False, diag=False):
    xs, ys, blinks = read_csv(path)
    valid = ~np.isnan(xs) & ~np.isnan(ys)
    if valid.sum() == 0:
        raise RuntimeError('No valid gaze points found in CSV')

    xs_valid = xs[valid]
    ys_valid = ys[valid]

    # Detect normalized coordinates in [0,1]
    is_normalized = (
        xs_valid.size > 0
        and xs_valid.max() <= 1.0
        and xs_valid.min() >= 0.0
        and ys_valid.max() <= 1.0
        and ys_valid.min() >= 0.0
    )

    # Get screen size if needed
    screen_w, screen_h = None, None
    if is_normalized or overlay:
        if pyautogui is not None:
            try:
                screen_w, screen_h = pyautogui.size()
            except Exception:
                screen_w, screen_h = None, None

    # If normalized, convert to pixels for plotting/overlay
    if is_normalized and screen_w is not None:
        xs_valid = xs_valid * screen_w
        ys_valid = ys_valid * screen_h
        xs = xs.copy(); ys = ys.copy()
        xs[valid] = xs_valid; ys[valid] = ys_valid

    if diag:
        print(f"Points total: {len(xs)}; valid: {valid.sum()}")
        print(f"x min/max: {np.nanmin(xs)}/{np.nanmax(xs)}")
        print(f"y min/max: {np.nanmin(ys)}/{np.nanmax(ys)}")
        print(f"Detected normalized coords: {is_normalized}")
        if screen_w is not None:
            print(f"Screen size detected: {screen_w} x {screen_h}")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Optional overlay screenshot
    if overlay:
        if screen_w is None or screen_h is None:
            print("Warning: cannot get screen size for overlay; skipping background screenshot")
        else:
            try:
                shot = pyautogui.screenshot()
                img = np.asarray(shot)
                # matplotlib expects RGB
                if img.shape[2] == 3:
                    ax.imshow(img, extent=[0, screen_w, screen_h, 0])
                else:
                    ax.imshow(img, extent=[0, screen_w, screen_h, 0])
            except Exception as e:
                print(f"Could not capture screenshot for overlay: {e}")

    ax.scatter(xs[valid], ys[valid], c='tab:blue', s=8, label='gaze')

    # Mark blinks (if any) as red X markers
    blink_idx = valid & (blinks == 1)
    if blink_idx.sum() > 0:
        ax.scatter(xs[blink_idx], ys[blink_idx], c='red', s=40, marker='x', label='blink')

    # If we used imshow with reversed y extent, don't invert; otherwise invert to match screen coordinates
    if not overlay:
        ax.invert_yaxis()

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title('Gaze locations')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if out_image:
        fig.savefig(out_image, dpi=150)
        print(f"Saved plot to {out_image}")

    if show:
        plt.show()


def cli():
    parser = argparse.ArgumentParser(description='Plot gaze CSV (defaults to webcam_eye/gaze_log.csv)')
    parser.add_argument('csv', nargs='?', default='webcam_eye/gaze_log.csv', help='Path to gaze CSV (default: webcam_eye/gaze_log.csv)')
    parser.add_argument('--out', '-o', help='Save image to path (optional)')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Do not show interactive window')
    parser.add_argument('--overlay', action='store_true', help='Overlay gaze points on a desktop screenshot (requires pyautogui)')
    parser.add_argument('--diag', action='store_true', help='Print diagnostics (min/max, counts, normalized detection)')
    args = parser.parse_args()
    plot(args.csv, out_image=args.out, show=args.show, overlay=args.overlay, diag=args.diag)


if __name__ == '__main__':
    cli()
