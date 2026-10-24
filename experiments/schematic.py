"""Figure 1: the experimental setup as a diagram.  python -m experiments.schematic

Coordinates are in inches (the axes span the whole figure), so text sizes and box sizes
stay consistent with each other.
"""
import os

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import BLUE, INK, INK2, LLM, MUTED, ORANGE, RL, plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures", "setup.png")
W, H = 10.0, 3.5


def box(ax, x0, y0, x1, y1, edge, fill="#ffffff", lw=1.3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=lw))


def text(ax, x, y, s, size=8, color=INK2, weight="normal", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=kw.pop("ha", "center"),
            va=kw.pop("va", "center"), linespacing=1.4, **kw)


def arrow(ax, a, b, color=MUTED, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.3,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # agents
    for y0, name in [(2.0, "Firm A"), (0.25, "Firm B")]:
        box(ax, 0.1, y0, 2.6, y0 + 1.25, RL)
        text(ax, 1.35, y0 + 0.95, f"{name}'s pricing agent", 9.5, INK, "semibold")
        text(ax, 1.35, y0 + 0.42, "Q-learner (Calvano et al. 2020)\nor LLM agent (Qwen2.5-7B)\n"
             "goal: its own long-run profit", 7.6)

    # observation filter
    box(ax, 3.05, 0.95, 4.45, 2.55, ORANGE, fill="#fff5ef")
    text(ax, 3.75, 2.27, "Intervention", 9, INK, "semibold")
    text(ax, 3.75, 1.6, "what each agent sees\nof its rival's price:\nexact · higher/lower\n"
         "noisy · delayed\nhidden · + messages", 7.4)

    # market
    box(ax, 5.0, 1.15, 6.5, 2.35, INK2)
    text(ax, 5.75, 2.03, "Market", 9.5, INK, "semibold")
    text(ax, 5.75, 1.55, "logit demand\nprofits → firms", 7.6)

    # auditor
    box(ax, 7.1, 0.25, 9.9, 3.25, BLUE)
    text(ax, 8.5, 3.0, "Auditor", 9.5, INK, "semibold")
    text(ax, 8.5, 2.62, "What agents do", 8.2, BLUE, "semibold")
    text(ax, 8.5, 2.08, "price level · parallel pricing\nlead–lag reaction\n"
         "forced price cut: does the rival retaliate?", 7.4)
    ax.plot([7.35, 9.65], [1.6, 1.6], color="#e6e5e1", lw=0.8)
    text(ax, 8.5, 1.35, "What agents say (LLMs)", 8.2, LLM, "semibold")
    text(ax, 8.5, 0.98, "keyword lexicon · LLM judge\nof the agent's private notes", 7.4)
    text(ax, 8.5, 0.48, "ground truth: blind or myopic agents cannot\ncollude, yet may still price high",
         6.8, MUTED, style="italic")

    # flows
    arrow(ax, (2.6, 3.0), (5.0, 2.3), rad=-0.22)
    text(ax, 3.75, 3.05, "price", 7.4, MUTED)
    arrow(ax, (2.6, 0.5), (5.0, 1.2), rad=0.22)
    text(ax, 3.75, 0.42, "price", 7.4, MUTED)
    arrow(ax, (5.0, 1.75), (4.45, 1.75), color=ORANGE)
    arrow(ax, (3.05, 2.05), (2.6, 2.3), color=ORANGE)
    arrow(ax, (3.05, 1.45), (2.6, 1.2), color=ORANGE)
    arrow(ax, (6.5, 1.75), (7.1, 1.75), color=BLUE)
    fig.savefig(OUT, bbox_inches=None, pad_inches=0)
    print(OUT)


if __name__ == "__main__":
    main()
