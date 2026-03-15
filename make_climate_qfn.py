"""Quantum field network climate visualisation — seamless loop rewrite."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_FRAMES    = 80
W, H        = 200, 120
N_NODES     = 18
N_SOURCES   = 4
N_PARTICLES = 30
EDGE_DIST   = 60
TAIL_LEN    = 8
PART_SPEED  = 3.0
RING_SPEED  = 4.0
MAX_RADIUS  = 80
RING_SPACING = 28

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Colour palette (unchanged)
# ---------------------------------------------------------------------------
cmap = mcolors.LinearSegmentedColormap.from_list(
    "climate_qfn",
    ["#2D0057", "#7B2FBE", "#FF6EC7", "#FFB3D9", "#FFF0F5"],
    N=256,
)

# ---------------------------------------------------------------------------
# Base field — 6 sine waves, seamless-loop phases
#
# Large scale (indices 0-2): low freq 0.02-0.04, amplitude 2.0, slow (1x)
# Small scale (indices 3-5): higher freq 0.06-0.10, amplitude 1.0, fast (2-3x)
# Speed multipliers (1,1,2,2,3,3) guarantee whole cycles in N_FRAMES.
# ---------------------------------------------------------------------------
LARGE_FREQ_RANGE = (0.02, 0.04)
SMALL_FREQ_RANGE = (0.06, 0.10)

freqs = np.concatenate([
    rng.uniform(*LARGE_FREQ_RANGE, (3, 2)),
    rng.uniform(*SMALL_FREQ_RANGE, (3, 2)),
])
amplitudes    = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
phase_offsets = rng.uniform(0, 2 * np.pi, (6, 2))
speed_mults   = np.array([1, 1, 2, 2, 3, 3])

x = np.linspace(0, W, W)
y = np.linspace(0, H, H)
X, Y = np.meshgrid(x, y)           # (H, W)


def compute_field(frame: int) -> np.ndarray:
    field = np.zeros((H, W))
    for k in range(6):
        t = 2 * np.pi * frame / N_FRAMES * speed_mults[k]
        field += amplitudes[k] * (
            np.sin(freqs[k, 0] * X + phase_offsets[k, 0] + t) *
            np.sin(freqs[k, 1] * Y + phase_offsets[k, 1] + t)
        )
    return field


# ---------------------------------------------------------------------------
# Bilinear interpolation helper — sample a 2D array at float (x, y) coords
# ---------------------------------------------------------------------------
def bilinear(arr: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    fx = px - np.floor(px)
    fy = py - np.floor(py)
    return (arr[y0, x0] * (1 - fx) * (1 - fy) +
            arr[y0, x1] * fx       * (1 - fy) +
            arr[y1, x0] * (1 - fx) * fy       +
            arr[y1, x1] * fx       * fy)


# ---------------------------------------------------------------------------
# Network nodes — jittered 6×3 grid
# ---------------------------------------------------------------------------
gx = np.linspace(18, W - 18, 6)
gy = np.linspace(18, H - 18, 3)
gX, gY = np.meshgrid(gx, gy)
nodes = np.clip(
    np.column_stack([gX.ravel(), gY.ravel()]) + rng.uniform(-14, 14, (N_NODES, 2)),
    [5, 5], [W - 5, H - 5],
)
node_phase = rng.uniform(0, 2 * np.pi, N_NODES)

edges = [
    (i, j)
    for i in range(N_NODES)
    for j in range(i + 1, N_NODES)
    if np.hypot(nodes[i, 0] - nodes[j, 0], nodes[i, 1] - nodes[j, 1]) < EDGE_DIST
]

# ---------------------------------------------------------------------------
# Interference ring sources — staggered phase offsets so rings are never
# all at the same expansion stage simultaneously
# ---------------------------------------------------------------------------
src = np.column_stack([
    rng.uniform(25, W - 25, N_SOURCES),
    rng.uniform(20, H - 20, N_SOURCES),
])
src_phase_offset = np.arange(N_SOURCES) * (MAX_RADIUS / N_SOURCES)

# ---------------------------------------------------------------------------
# Particles — positions + tail history
# ---------------------------------------------------------------------------
part_x = rng.uniform(0, W, N_PARTICLES)
part_y = rng.uniform(0, H, N_PARTICLES)
# history[n, 0] = most recent position, history[n, -1] = oldest
part_history = np.full((N_PARTICLES, TAIL_LEN, 2), np.nan)


def respawn(n: int) -> None:
    """Respawn particle n at a random canvas edge, clearing its tail."""
    edge = rng.integers(0, 4)
    if edge == 0:
        part_x[n], part_y[n] = 0.0,          rng.uniform(0, H)
    elif edge == 1:
        part_x[n], part_y[n] = float(W - 1), rng.uniform(0, H)
    elif edge == 2:
        part_x[n], part_y[n] = rng.uniform(0, W), 0.0
    else:
        part_x[n], part_y[n] = rng.uniform(0, W), float(H - 1)
    part_history[n] = np.nan


# ---------------------------------------------------------------------------
# Transparency export helper
# ---------------------------------------------------------------------------
def _rgba_to_gif_frame(img: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    p = bg.convert("RGB").quantize(colors=255, method=Image.Quantize.FASTOCTREE)
    alpha   = img.split()[3]
    mask    = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
    p_data  = list(p.getdata())
    mk_data = list(mask.getdata())
    p.putdata([255 if mk_data[j] == 255 else p_data[j] for j in range(len(p_data))])
    p.info["transparency"] = 255
    return p


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------
frames_pil = []

for frame in range(N_FRAMES):

    # ---- field + gradient --------------------------------------------------
    field  = compute_field(frame)
    grad_y, grad_x = np.gradient(field)     # velocity components

    # ---- move particles (bilinear-interpolated gradient) -------------------
    vx  = bilinear(grad_x, part_x, part_y)
    vy  = bilinear(grad_y, part_x, part_y)
    mag = np.hypot(vx, vy) + 1e-8
    new_x = part_x + PART_SPEED * vx / mag
    new_y = part_y + PART_SPEED * vy / mag

    # Respawn particles that left the canvas
    out = (new_x < 0) | (new_x >= W) | (new_y < 0) | (new_y >= H)
    part_x[:] = new_x
    part_y[:] = new_y
    for n in np.where(out)[0]:
        respawn(n)

    # Update tail history: shift older entries back, store current at [0]
    part_history[:, 1:, :] = part_history[:, :-1, :]
    part_history[:, 0, 0]  = part_x
    part_history[:, 0, 1]  = part_y

    # ---- draw --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

    # base field
    ax.contourf(X, Y, field, levels=40, cmap=cmap, alpha=0.85, zorder=1)

    # interference rings
    for s in range(N_SOURCES):
        for ring_idx in range(3):
            r = (frame * RING_SPEED + src_phase_offset[s] +
                 ring_idx * RING_SPACING) % MAX_RADIUS
            if r < 4:
                continue
            alpha = max(0.0, 0.6 - r / 80.0)
            ax.add_patch(Circle(
                (src[s, 0], src[s, 1]), r,
                fill=False, edgecolor="#FF6EC7",
                linewidth=1.0, alpha=alpha, zorder=3,
            ))

    # network edges
    for i, j in edges:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            color="#CC99FF", linewidth=0.8, alpha=0.4, zorder=4,
        )

    # network nodes
    for n in range(N_NODES):
        pulse_r = 4.5 + 1.5 * np.sin(frame * 0.25 + node_phase[n])
        ax.scatter(nodes[n, 0], nodes[n, 1],
                   s=pulse_r ** 2 * 5, color="#FF6EC7",
                   alpha=0.3, linewidths=0, zorder=5)
        ax.scatter(nodes[n, 0], nodes[n, 1],
                   s=15, color="white", alpha=1.0,
                   linewidths=0, zorder=6)

    # particle tails — batch as a single LineCollection
    tail_segs   = []
    tail_colors = []
    r_gold, g_gold, b_gold = 1.0, 0.843, 0.0   # #FFD700
    for n in range(N_PARTICLES):
        hist  = part_history[n]                 # (TAIL_LEN, 2), [0]=newest
        valid = ~np.isnan(hist[:, 0])
        vh    = hist[valid]
        k     = len(vh)
        if k < 2:
            continue
        for seg_i in range(k - 1):
            # seg_i=0 connects newest→2nd newest (most opaque end)
            alpha_seg = (1.0 - seg_i / (k - 1)) * 0.7
            tail_segs.append([vh[seg_i], vh[seg_i + 1]])
            tail_colors.append([r_gold, g_gold, b_gold, alpha_seg])

    if tail_segs:
        ax.add_collection(LineCollection(
            tail_segs, colors=tail_colors, linewidths=1.2, zorder=7,
        ))

    # particle dots
    ax.scatter(part_x, part_y, s=8, color="#FFD700",
               alpha=0.7, linewidths=0, zorder=8)

    # capture
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w_px, h_px), fig.canvas.buffer_rgba())
    frames_pil.append(img)
    plt.close(fig)

    if (frame + 1) % 20 == 0:
        print(f"  rendered {frame + 1}/{N_FRAMES} frames")

# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------
gif_frames = [_rgba_to_gif_frame(f) for f in frames_pil]

gif_frames[0].save(
    "climate_qfn.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=50,
    loop=0,
    optimize=False,
    transparency=255,
    disposal=2,
)
print("Saved climate_qfn.gif")
