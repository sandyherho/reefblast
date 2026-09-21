"""Dark style, signed colour map, and looping GIF export for animations.

Animations are rendered frame by frame with Matplotlib into RGB arrays and
written as palette GIFs with Pillow.  Every animation is a direct render of
a computed model field; nothing is interpolated for visual effect beyond
the display nonlinearity documented in each script.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

matplotlib.use("Agg")

__all__ = ["dark", "SIGNED", "signed_power", "fig_to_rgb", "write_gif",
           "ANIMDIR"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

BG = "#07090f"
FG = "#d9dee8"

# tension deep blue to cyan, zero near black, compression amber to white
SIGNED = LinearSegmentedColormap.from_list("signed", [
    (0.00, "#e8f7ff"), (0.18, "#35c3ff"), (0.36, "#1446a0"),
    (0.50, BG), (0.64, "#8a1c0a"), (0.82, "#ff8a1f"), (1.00, "#fff4d6"),
])


def dark():
    """Apply the animation style."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#3a4152",
        "xtick.color": FG, "ytick.color": FG, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.6,
    })


def signed_power(x, scale, power=0.5):
    """Sign-preserving power law, mapped to [-1, 1] at |x| = scale."""
    y = np.sign(x) * (np.abs(x) / scale) ** power
    return np.clip(y, -1.0, 1.0)


def fig_to_rgb(fig):
    """Rasterise a figure to an (h, w, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def write_gif(stem, frames, fps=18, colors=192):
    """Write a looping, palette-quantised GIF and return its path."""
    os.makedirs(ANIMDIR, exist_ok=True)
    path = os.path.join(ANIMDIR, f"{stem}.gif")
    imgs = [Image.fromarray(f).quantize(colors=colors, method=Image.MEDIANCUT,
                                        dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(round(1000 / fps)), optimize=True, disposal=2)
    return path
