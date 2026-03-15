"""Generate an animated Bloch sphere GIF using matplotlib, numpy, and Pillow."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_FRAMES   = 120
TRAIL_LEN  = 40
WIREFRAME_COLOR = "#6A0DAD"
CIRCLE_COLOR    = "#7B2FBE"
MAGENTA         = "#FF6EC7"
TRAIL_END_COLOR = "#CC99FF"
AXIS_LINE_COLOR = "#666666"
AXIS_LABEL_COLOR = "#444444"
POLE_DOT_COLOR  = "#6A0DAD"
LABEL_COLOR     = "#2D0057"
THETA_FIXED = np.pi / 3          # fixed polar angle of state vector

# ---------------------------------------------------------------------------
# Sphere geometry helpers
# ---------------------------------------------------------------------------

def _sphere_wireframe(n=20):
    """Return lists of (x,y,z) line segments for latitude and longitude lines."""
    segs = []
    u = np.linspace(0, 2 * np.pi, 120)

    # latitude rings
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n):
        x = np.cos(lat) * np.cos(u)
        y = np.cos(lat) * np.sin(u)
        z = np.sin(lat) * np.ones_like(u)
        segs.append(list(zip(x, y, z)))

    # longitude arcs
    v = np.linspace(0, np.pi, 120)
    for lon in np.linspace(0, 2 * np.pi, n, endpoint=False):
        x = np.sin(v) * np.cos(lon)
        y = np.sin(v) * np.sin(lon)
        z = np.cos(v)
        segs.append(list(zip(x, y, z)))

    return segs


def _circle_xz(n=300):
    """Great circle in the xz plane."""
    t = np.linspace(0, 2 * np.pi, n)
    return np.cos(t), np.zeros(n), np.sin(t)


def _equator(n=300):
    """Equatorial circle."""
    t = np.linspace(0, 2 * np.pi, n)
    return np.cos(t), np.sin(t), np.zeros(n)


def _tip(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


# ---------------------------------------------------------------------------
# Pre-compute all tip positions
# ---------------------------------------------------------------------------
phis = 2 * np.pi * np.arange(N_FRAMES) / N_FRAMES
tips = np.array([_tip(THETA_FIXED, phi) for phi in phis])  # (N_FRAMES, 3)

wireframe_segs = _sphere_wireframe(20)

# ---------------------------------------------------------------------------
# Render frames
# ---------------------------------------------------------------------------
frames_pil = []

for i in range(N_FRAMES):
    fig = plt.figure(figsize=(6, 6), dpi=100)
    fig.patch.set_alpha(0.0)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0.0)

    # ---- remove box / panes ------------------------------------------------
    ax.set_axis_off()
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])

    # ---- wireframe sphere --------------------------------------------------
    col = Line3DCollection(
        wireframe_segs,
        colors=WIREFRAME_COLOR, alpha=0.45, linewidths=1.0, zorder=1,
    )
    ax.add_collection3d(col)

    # ---- equator (dashed) --------------------------------------------------
    ex, ey, ez = _equator()
    ax.plot(ex, ey, ez, color=CIRCLE_COLOR, alpha=0.7, linewidth=1.5,
            linestyle="--", zorder=2)

    # ---- xz great circle (dashed) ------------------------------------------
    gx, gy, gz = _circle_xz()
    ax.plot(gx, gy, gz, color=CIRCLE_COLOR, alpha=0.7, linewidth=1.5,
            linestyle="--", zorder=2)

    # ---- axis lines + labels -----------------------------------------------
    L = 1.35
    for (dx, dy, dz), label, ha, va in [
        ((L, 0, 0), "x", "left",   "center"),
        ((0, L, 0), "y", "left",   "center"),
        ((0, 0, L), "z", "center", "bottom"),
    ]:
        ax.plot([0, dx], [0, dy], [0, dz], color=AXIS_LINE_COLOR, linewidth=1, zorder=2)
        ax.text(dx, dy, dz, label, color=AXIS_LABEL_COLOR, fontsize=11,
                ha=ha, va=va)

    # ---- poles -------------------------------------------------------------
    ax.scatter([0], [0], [1],  color=POLE_DOT_COLOR, s=40, zorder=5)
    ax.scatter([0], [0], [-1], color=POLE_DOT_COLOR, s=40, zorder=5)
    ax.text(0.07,  0, 1.08,  "|0⟩", color=LABEL_COLOR, fontsize=12,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.07, 0, -1.13, "|1⟩", color=LABEL_COLOR, fontsize=12,
            fontweight="bold", ha="left", va="top")

    # ---- trail -------------------------------------------------------------
    start = max(0, i - TRAIL_LEN + 1)
    trail = tips[start : i + 1]           # (k, 3)
    k = len(trail)
    if k > 1:
        # build segments with per-segment alpha fading old→new
        seg_list = [
            [trail[j], trail[j + 1]] for j in range(k - 1)
        ]
        alphas = np.linspace(0.0, 1.0, k - 1)
        # Interpolate RGB from TRAIL_END_COLOR (#CC99FF) → MAGENTA (#FF6EC7)
        r0, g0, b0 = 0.800, 0.600, 1.000   # #CC99FF
        r1, g1, b1 = 1.000, 0.432, 0.780   # #FF6EC7
        t = np.linspace(0.0, 1.0, k - 1)
        colors = np.zeros((k - 1, 4))
        colors[:, 0] = r0 + (r1 - r0) * t
        colors[:, 1] = g0 + (g1 - g0) * t
        colors[:, 2] = b0 + (b1 - b0) * t
        colors[:, 3] = alphas
        trail_col = Line3DCollection(seg_list, colors=colors,
                                     linewidths=2.5, zorder=4)
        ax.add_collection3d(trail_col)

    # ---- state vector arrow ------------------------------------------------
    tx, ty, tz = tips[i]
    ax.quiver(0, 0, 0, tx, ty, tz,
              color='#FF6EC7', linewidth=3,
              arrow_length_ratio=0.15, normalize=False, zorder=6)

    # ---- consistent view angle ---------------------------------------------
    ax.view_init(elev=20, azim=35)

    # ---- capture frame -----------------------------------------------------
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", (w, h), buf)
    frames_pil.append(img)
    plt.close(fig)

    if (i + 1) % 20 == 0:
        print(f"  rendered {i + 1}/{N_FRAMES} frames")

# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------
# Convert RGBA frames to P mode with transparency for GIF export.
# GIF supports a single transparent colour index; we map the alpha
# channel by quantising to a palette and marking the background index.
def _rgba_to_gif_frame(img: Image.Image) -> Image.Image:
    # Paste onto a white canvas to get a clean palette quantisation,
    # then re-apply transparency via the alpha channel as a binary mask.
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    p = background.convert("RGB").quantize(colors=255, method=Image.Quantize.FASTOCTREE)
    # Add transparent index (255) for fully-transparent pixels
    alpha = img.split()[3]
    mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
    p_data = list(p.getdata())
    mask_data = list(mask.getdata())
    p_data = [255 if mask_data[j] == 255 else p_data[j] for j in range(len(p_data))]
    p.putdata(p_data)
    p.info["transparency"] = 255
    return p

gif_frames = [_rgba_to_gif_frame(f) for f in frames_pil]

gif_frames[0].save(
    "bloch_sphere.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=33,
    loop=0,
    optimize=False,
    transparency=255,
    disposal=2,
)
print("Saved bloch_sphere.gif")
