"""Shared figure style for every chart in the paper."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
RL, LLM = BLUE, ORANGE          # agent families keep the same colour in every figure

plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "lines.linewidth": 1.8, "lines.markersize": 4.5,
})


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
