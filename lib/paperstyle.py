"""Shared style for every figure in the paper (ACM sigconf, two columns).

Figures are drawn at their printed size, so the font sizes below are the sizes on the page:
full width = 7.0in (\\textwidth), single column = 3.33in (\\columnwidth).
Import and call apply() before drawing; save() writes PDF and PNG at the exact figsize.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED = "#9a9892", "#1e1d1a", "#6b6a66"
LIGHTBLUE = "#9ec5f4"
PALEGREY = "#dcd9d0"        # neutral bar fill ("no effect", leader lines)
GRIDC = "#e6e4de"
EDGE = "#c9c7c0"

FULL, COL = 7.0, 3.33       # printed widths in inches
TICK, TEXT, LEG, TITLE = 6.5, 7, 6.5, 7.5   # tick labels, labels/annotations, legends, panel titles


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": TEXT,
        "axes.labelsize": TEXT, "axes.titlesize": TITLE, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlepad": 4,
        "axes.labelcolor": INK, "axes.labelpad": 2.5, "axes.edgecolor": EDGE, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRIDC, "grid.linewidth": 0.5,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.fontsize": LEG, "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "lines.linewidth": 1.2, "lines.markersize": 3,
        "text.color": INK, "patch.linewidth": 0,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
        "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def polish(ax, axis="y"):
    """Light grid on one axis only ('y' by default, 'x' for horizontal bars)."""
    ax.grid(False)
    if axis:
        ax.grid(axis=axis, color=GRIDC, lw=0.5)
    ax.set_axisbelow(True)


def figure(width, height, **kw):
    return plt.subplots(figsize=(width, height), layout="constrained", **kw)


def save(fig, stem):
    """Write stem.pdf and stem.png at exactly the figure size (no tight-bbox cropping)."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    _check_clipping(fig, stem)
    plt.close(fig)


def _check_clipping(fig, stem):
    """Warn when ink touches the edge of the saved PNG (a sign of clipped text)."""
    img = plt.imread(f"{stem}.png")[..., :3]
    dark = (img < 0.85).any(axis=2)
    edges = {"top": dark[0].any(), "bottom": dark[-1].any(), "left": dark[:, 0].any(), "right": dark[:, -1].any()}
    hit = [k for k, v in edges.items() if v]
    if hit:
        print(f"WARNING {stem}: ink touches the {', '.join(hit)} edge (clipped?)")


apply()
